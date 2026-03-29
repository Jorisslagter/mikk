import * as path from 'node:path'
import * as fs from 'node:fs/promises'
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'
import {
    ContractReader, LockReader,
    ImpactAnalyzer, DeadCodeDetector, AdrManager,
    BoundaryChecker,
    type MikkContract, type MikkLock,
    type DependencyGraph, type GraphNode, type GraphEdge,
    BM25Index, buildFunctionTokens, reciprocalRankFusion, tokenize,
} from '@getmikk/core'
import { ContextBuilder, getProvider } from '@getmikk/ai-context'
import { SemanticSearcher } from '@getmikk/intent-engine'
import type { ContextQuery } from '@getmikk/ai-context'


// Cache contract+lock+graph per project root with 30s TTL to avoid re-reading
// from disk on every MCP tool call (~200ms I/O saved per call)

interface CachedProject {
    contract: MikkContract
    lock: MikkLock
    graph: DependencyGraph
    staleness: string | null
    cachedAt: number
}

const projectCache = new Map<string, CachedProject>()
const CACHE_TTL_MS = 30_000 // 30 seconds

function invalidateCache(projectRoot: string): void {
    projectCache.delete(projectRoot)
}

// Semantic searcher singletons per project root.
// Capped at MAX_SEARCHER_ROOTS to prevent unbounded memory growth in
// long-running MCP sessions with many project roots.
const MAX_SEARCHER_ROOTS = 5
const MAX_QUERY_HOPS = 12
const MAX_QUERY_TOKEN_BUDGET = 20_000
const MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
const MAX_WALK_DIR_DEPTH = 10
const MAX_WALK_FILES = 10_000
function getSemanticSearcher(projectRoot: string): SemanticSearcher {
    let s = semanticSearchers.get(projectRoot)
    if (!s) {
        // Evict the oldest entry when the cap is exceeded
        if (semanticSearchers.size >= MAX_SEARCHER_ROOTS) {
            const oldestKey = semanticSearchers.keys().next().value
            if (oldestKey !== undefined) semanticSearchers.delete(oldestKey)
        }
        s = new SemanticSearcher(projectRoot)
        semanticSearchers.set(projectRoot, s)
    }
    return s
}
const _CPT = 4; const _ALC = 42
interface TokenTally { calls: number; used: number; raw: number; saved: number; start: number }
const _tallies = new Map<string, TokenTally>()
function _tally(r: string): TokenTally { let t = _tallies.get(r); if (!t) { t = { calls: 0, used: 0, raw: 0, saved: 0, start: Date.now() }; _tallies.set(r, t) } return t }
function _tok(o: unknown): number { return Math.max(1, Math.round(JSON.stringify(o).length / _CPT)) }
function _fileTok(lock: MikkLock, fp: string): number { const fs2 = Object.values(lock.functions).filter(f => f.file === fp); const ln = fs2.length > 0 ? Math.max(...fs2.map(f => f.endLine)) : 80; return Math.round((ln * _ALC) / _CPT) }
function _filesTok(lock: MikkLock, fps: string[]): number { return fps.reduce((s, f) => s + _fileTok(lock, f), 0) }
function _track(root: string, raw: number, resp: unknown): Record<string, number> {
    const used = _tok(resp); const saved = Math.max(0, raw - used); const t = _tally(root)
    t.calls++; t.used += used; t.raw += raw; t.saved += saved
    return { used, raw, saved, sessionSaved: t.saved, sessionCalls: t.calls }
}

// Singleton per projectRoot — pipeline load is ~1-2s, must not repeat per request
const semanticSearchers = new Map<string, SemanticSearcher>()

/** Quick-hash a file by reading first 8KB for fast drift detection */
async function quickHashFile(filePath: string): Promise<string> {
    let handle: Awaited<ReturnType<typeof fs.open>> | null = null
    try {
        handle = await fs.open(filePath, 'r')
        const buf = Buffer.alloc(8192)
        const { bytesRead } = await handle.read(buf, 0, 8192, 0)
        return createHash('sha256').update(buf.subarray(0, bytesRead)).digest('hex').slice(0, 16)
    } catch {
        return 'unreadable'
    } finally {
        if (handle) {
            try { await handle.close() } catch { /* best-effort close */ }
        }
    }
}

/**
 * Register all MCP tools — actions an AI assistant can invoke.
 */
export function registerTools(server: McpServer, projectRoot: string) {


    // TOOL: mikk_test_tool

    server.tool(
        'mikk_test_tool',
        'A simple test tool that returns a static message.',
        {},
        async () => {
            return { content: [{ type: 'text', text: 'Mikk test tool executed successfully.' }] }
        },
    )


    // TOOL: mikk_get_project_overview

    server.tool(
        'mikk_get_project_overview',
        'Get a high-level overview: modules, function counts, file counts, constraints. WHEN TO USE: When you need raw project stats. For session start, prefer mikk_get_session_context instead. AFTER THIS: Use mikk_query_context with your task, or mikk_list_modules to drill into a module.',
        {},
        async () => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)

            const modules = contract.declared.modules.map(mod => {
                const fns = Object.values(lock.functions).filter(f => f.moduleId === mod.id)
                const files = Object.values(lock.files).filter(f => f.moduleId === mod.id)
                return {
                    id: mod.id,
                    name: mod.name,
                    description: mod.description,
                    functions: fns.length,
                    files: files.length,
                    exported: fns.filter(f => f.isExported).length,
                }
            })

            const overview = {
                project: contract.project,
                totalFunctions: Object.keys(lock.functions).length,
                totalFiles: Object.keys(lock.files).length,
                totalModules: modules.length,
                modules,
                constraints: contract.declared.constraints,
                decisions: contract.declared.decisions,
                warning: staleness,
                hint: 'Next: Use mikk_query_context with your task description, or mikk_list_modules to explore the architecture.',
            }

            // Token savings: replaces agent reading every module's files to get project structure
            const _rawOverview = Math.min(15, Object.keys(lock.files).length) * Math.round((80 * _ALC) / _CPT)
                ; (overview as any).tokens = _track(projectRoot, _rawOverview, overview)
            return { content: [{ type: 'text' as const, text: JSON.stringify(overview, null, 2) }] }
        },
    )


    // TOOL: mikk_query_context

    server.tool(
        'mikk_query_context',
        'Ask an architecture question — returns graph-traced context with relevant functions, files, and call chains. Use this to understand how code flows through the project.',
        {
            question: z.string().describe('The architecture question or task description'),
            maxHops: z.number().int().min(1).max(MAX_QUERY_HOPS).optional().default(4).describe('Graph traversal depth (default: 4)'),
            tokenBudget: z.number().int().min(256).max(MAX_QUERY_TOKEN_BUDGET).optional().default(6000).describe('Max tokens for function bodies (default: 6000)'),
            focusFile: z.string().optional().describe('Anchor traversal from a specific file path'),
            focusModule: z.string().optional().describe('Anchor traversal from a specific module ID'),
            strict: z.boolean().optional().default(false).describe('High-precision mode: include only tightly relevant context'),
            requiredTerms: z.array(z.string()).optional().describe('Required terms that must match returned context'),
            requireAllKeywords: z.boolean().optional().default(false).describe('In strict mode, require all extracted keywords'),
            minKeywordMatches: z.number().optional().default(1).describe('In strict mode, minimum keyword hits per function'),
            exactOnly: z.boolean().optional().default(false).describe('Hard gate: keep only strict keyword matches'),
            failFast: z.boolean().optional().default(false).describe('Return no context if strict filters find no exact match'),
            autoFallback: z.boolean().optional().default(true).describe('When strict mode returns empty, retry with balanced retrieval'),
            provider: z.enum(['claude', 'generic', 'compact']).optional().default('generic').describe('AI provider format: claude (XML tags), generic (plain), compact (minimal tokens)'),
        },
        async ({ question, maxHops, tokenBudget, focusFile, focusModule, strict, requiredTerms, requireAllKeywords, minKeywordMatches, exactOnly, failFast, autoFallback, provider }) => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)

            const query: ContextQuery = {
                task: question,
                maxHops,
                tokenBudget,
                focusFiles: focusFile ? [focusFile] : undefined,
                focusModules: focusModule ? [focusModule] : undefined,
                includeCallGraph: true,
                includeBodies: true,
                relevanceMode: strict ? 'strict' : 'balanced',
                requiredKeywords: requiredTerms,
                requireAllKeywords,
                minKeywordMatches,
                exactOnly,
                failFast,
                projectRoot,
            }

            const builder = new ContextBuilder(contract, lock)
            let ctx = builder.build(query)
            let fallbackUsed = false
            if (autoFallback !== false && strict && ctx.modules.length === 0) {
                const relaxed: ContextQuery = {
                    ...query,
                    relevanceMode: 'balanced',
                    requiredKeywords: undefined,
                    requireAllKeywords: false,
                    minKeywordMatches: 1,
                    exactOnly: false,
                    failFast: false,
                }
                const fallback = builder.build(relaxed)
                if (fallback.modules.length > 0) {
                    ctx = fallback
                    fallbackUsed = true
                    ctx.meta.reasons = [
                        ...(ctx.meta.reasons ?? []),
                        'strict query had no exact matches; returned balanced fallback context',
                    ]
                }
            }

            if (ctx.modules.length === 0) {
                return {
                    content: [{
                        type: 'text' as const,
                        text: `No context found for "${question}". ${focusFile
                            ? `The file "${focusFile}" may not exist in the lock.`
                            : 'The project may have no analyzed functions.'
                            } Run \`mikk analyze\` or check the file path.`,
                    }],
                    isError: true,
                }
            }

            const formatter = getProvider(provider ?? 'generic')
            const output = formatter.formatContext(ctx)
            const warning = staleness ? `\n\n${staleness}` : ''
            const fallbackNote = fallbackUsed
                ? 'Note: strict mode had no exact matches; showing balanced fallback context.\n\n'
                : ''

            // Token savings: tokenBudget is the cap — raw cost without Mikk is reading all files naively
            const _rawQC = (tokenBudget ?? 6000) * 3   // Mikk's BFS gives ~3x compression over naive search
            const _tokQC = _track(projectRoot, _rawQC, output)
            const tokLine = `\n\n---\n// tokens: ${JSON.stringify(_tokQC)}`
            return {
                content: [{ type: 'text' as const, text: fallbackNote + output + warning + '\n\n---\nHint: Use mikk_before_edit on any files you plan to modify, then mikk_impact_analysis to see the full blast radius.' + tokLine }],
            }
        },
    )


    // TOOL: mikk_impact_analysis

    server.tool(
        'mikk_impact_analysis',
        'Analyze the blast radius of changing a file. Returns impacted functions classified by severity (critical/high/medium/low). WHEN TO USE: Before refactoring, renaming, or modifying shared code. AFTER THIS: Use mikk_get_function_detail on critical/high items to review them.',
        {
            file: z.string().describe('The file path (relative to project root) to analyze impact for'),
        },
        async ({ file }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)
            const graph = buildGraphFromLock(lock)
            const analyzer = new ImpactAnalyzer(graph)

            const normalizedFile = file.replace(/\\/g, '/')
            let fileNodes = [...graph.nodes.values()].filter(n => n.file === normalizedFile)

            if (fileNodes.length === 0) {
                const basename = normalizedFile.split('/').pop() || normalizedFile
                fileNodes = [...graph.nodes.values()].filter(n => {
                    const nodeName = n.file.split('/').pop() || n.file
                    return nodeName === basename
                })
            }

            if (fileNodes.length === 0) {
                return {
                    content: [{ type: 'text' as const, text: `No functions found in "${file}". Use mikk_search_functions to look up the correct path, or mikk_list_modules to explore by module.` }],
                    isError: true,
                }
            }

            const result = analyzer.analyze(fileNodes.map(n => n.id))

            const impactedDetails = result.impacted.slice(0, 30).map(id => {
                const node = graph.nodes.get(id)
                return { function: node?.name ?? id, file: node?.file ?? '', module: node?.moduleId ?? '' }
            })

            const response = {
                file,
                changedNodes: result.changed.length,
                impactedNodes: result.impacted.length,
                depth: result.depth,
                confidence: result.confidence,
                classified: {
                    critical: result.classified.critical.length,
                    high: result.classified.high.length,
                    medium: result.classified.medium.length,
                    low: result.classified.low.length,
                    criticalItems: result.classified.critical.slice(0, 10),
                    highItems: result.classified.high.slice(0, 10),
                },
                impacted: impactedDetails,
                truncated: result.impacted.length > 30,
                warning: staleness,
                hint: 'Next: Use mikk_get_function_detail on critical/high items to review them. Then mikk_before_edit to validate your planned changes.',
            }

            // Token savings: replaces reading the changed file + all its dependents manually
            const _rawIA = _fileTok(lock, normalizedFile) + result.impacted.length * Math.round((40 * _ALC) / _CPT)
                ; (response as any).tokens = _track(projectRoot, _rawIA, response)
            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_search_functions

    server.tool(
        'mikk_search_functions',
        'Search for functions by name or ID using a hybrid BM25+substring search. WHEN TO USE: When you need to find a function but are unsure of its exact name or location. AFTER THIS: Use mikk_get_function_detail to get more information about a specific function.',
        {
            query: z.string().describe('The search query for function names or IDs'),
            limit: z.number().optional().default(10).describe('Maximum number of results to return'),
        },
        async ({ query, limit }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)
            const allFunctions = Object.values(lock.functions)
            const queryLower = query.toLowerCase()

            // --- Substring matches (fast, deterministic) ---
            const substringMatches = allFunctions
                .filter(fn => fn.name.toLowerCase().includes(queryLower) || fn.id.toLowerCase().includes(queryLower))
                .map((fn, i) => ({ id: fn.id, score: 100 - i }))

            // --- BM25 matches (ranked by relevance) ---
            const bm25 = new BM25Index()
            for (const fn of allFunctions) {
                bm25.addDocument(fn.id, buildFunctionTokens(fn))
            }
            const bm25Matches = bm25.search(query, limit * 2)

            // --- Reciprocal Rank Fusion to merge both lists ---
            const fused = reciprocalRankFusion(substringMatches, bm25Matches)

            const matches = fused
                .slice(0, limit)
                .map(result => {
                    const fn = lock.functions[result.id]
                    if (!fn) return null
                    return {
                        name: fn.name,
                        file: fn.file,
                        module: fn.moduleId,
                        exported: fn.isExported,
                        lines: `${fn.startLine}-${fn.endLine}`,
                        relevance: Math.round(result.score * 10000) / 10000,
                    }
                })
                .filter(Boolean)

            if (matches.length === 0) {
                return { content: [{ type: 'text' as const, text: `No functions matching "${query}" found.` }] }
            }

            const response = {
                matches,
                searchMethod: 'hybrid (BM25 + substring via RRF)',
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_before_edit

    server.tool(
        'mikk_before_edit',
        'MANDATORY: Call BEFORE editing any file. Returns blast radius, exported functions at risk, constraint violations (6 rule types), and circular dependency warnings. WHEN TO USE: ALWAYS before modifying files. AFTER THIS: If constraintStatus is fail, redesign your approach. If pass, proceed with edits. TIP: Pass multiple files for combined blast radius.',
        {
            files: z.array(z.string()).min(1).max(20).describe('The file paths (relative to project root) you are about to edit'),
        },
        async ({ files: filesToEdit }) => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)
            const graph = buildGraphFromLock(lock)
            const analyzer = new ImpactAnalyzer(graph)

            // Run boundary checker to detect actual constraint violations
            const checker = new BoundaryChecker(contract, lock)
            const boundaryResult = checker.check()

            const fileReports: Record<string, any> = {}

            for (const file of filesToEdit) {
                const normalizedFile = file.replace(/\\/g, '/').replace(/^\.\//, '')

                const fileFns = Object.values(lock.functions).filter(
                    fn => fn.file === normalizedFile || fn.file.endsWith('/' + normalizedFile),
                )

                if (fileFns.length === 0) {
                    fileReports[file] = {
                        warning: 'No tracked functions found in this file. Run `mikk analyze` to update the lock, or use mikk_search_functions to verify the file path.',
                    }
                    continue
                }

                const result = analyzer.analyze(fileFns.map(fn => fn.id))
                const impactedDetails = result.impacted.slice(0, 20).map(id => {
                    const node = graph.nodes.get(id)
                    return { function: node?.name ?? id, file: node?.file ?? '', module: node?.moduleId ?? '' }
                })

                const exportedAtRisk = fileFns.filter(fn => fn.isExported).map(fn => ({
                    name: fn.name,
                    calledBy: fn.calledBy.map(id => lock.functions[id]?.name).filter(Boolean),
                }))

                // Filter violations relevant to this file
                const fileViolations = boundaryResult.violations.filter(
                    v => v.from.file === normalizedFile || v.from.file.endsWith('/' + normalizedFile)
                ).map(v => ({
                    type: 'boundary_violation',
                    severity: v.severity,
                    rule: v.rule,
                    from: `${v.from.moduleName}::${v.from.functionName}`,
                    to: `${v.to.moduleName}::${v.to.functionName}`,
                    message: `${v.from.moduleName}::${v.from.functionName} -> ${v.to.moduleName}::${v.to.functionName} violates: "${v.rule}"`,
                }))

                // Detect circular dependencies for this file's functions
                const circularWarnings = detectCircularDeps(fileFns, lock)

                fileReports[file] = {
                    functionsInFile: fileFns.map(fn => fn.name),
                    exportedAtRisk,
                    impactedNodes: result.impacted.length,
                    depth: result.depth,
                    confidence: result.confidence,
                    impacted: impactedDetails,
                    truncated: result.impacted.length > 20,
                    constraints: contract.declared.constraints,
                    constraintStatus: fileViolations.length === 0 ? 'pass' : 'fail',
                    violations: fileViolations,
                    circularDependencies: circularWarnings,
                }
            }

            const totalImpact = Object.values(fileReports)
                .filter(r => typeof r.impactedNodes === 'number')
                .reduce((sum, r) => sum + r.impactedNodes, 0)

            const totalViolations = Object.values(fileReports)
                .reduce((sum, r) => sum + (r.violations?.length ?? 0), 0)

            const response = {
                summary: `Editing ${filesToEdit.length} file(s). Blast radius: ${totalImpact} dependent node(s). Constraint violations: ${totalViolations}.`,
                constraintStatus: totalViolations === 0 ? 'pass' : 'fail',
                files: fileReports,
                warning: staleness,
                hint: totalViolations > 0
                    ? '⚠ Constraint violations detected! Review the violations before proceeding. Use mikk_get_constraints for full rule context.'
                    : 'All constraints satisfied. If safe, proceed with your edits.',
            }

            // Token savings: replaces reading each edited file + tracing call graph manually
            const _rawBE = _filesTok(lock, filesToEdit) * 4  // file contents + dependency traversal
                ; (response as any).tokens = _track(projectRoot, _rawBE, response)
            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_list_modules

    server.tool(
        'mikk_list_modules',
        'List all declared modules with file counts, function counts, entry points, and descriptions. WHEN TO USE: To explore the project structure. Good starting point after mikk_get_session_context. AFTER THIS: Use mikk_get_module_detail with a specific moduleId.',
        {},
        async () => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)

            const modules = contract.declared.modules.map(mod => {
                const fns = Object.values(lock.functions).filter(f => f.moduleId === mod.id)
                const files = Object.values(lock.files).filter(f => f.moduleId === mod.id)
                return {
                    id: mod.id,
                    name: mod.name,
                    description: mod.description,
                    paths: mod.paths,
                    functions: fns.length,
                    files: files.length,
                    entryFunctions: mod.entryFunctions ?? [],
                }
            })

            const response = {
                modules,
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_get_module_detail

    server.tool(
        'mikk_get_module_detail',
        'Deep dive into a single module: all functions, files, exported API surface, internal call graph. WHEN TO USE: After mikk_list_modules to understand a specific module. AFTER THIS: Use mikk_get_function_detail for specific functions, or mikk_before_edit if modifying files in this module.',
        {
            moduleId: z.string().describe('The module ID (e.g., "packages-core", "lib-auth")'),
        },
        async ({ moduleId }) => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)
            const mod = contract.declared.modules.find(m => m.id === moduleId)

            if (!mod) {
                return {
                    content: [{ type: 'text' as const, text: `Module "${moduleId}" not found. Use mikk_list_modules to see available modules.` }],
                    isError: true,
                }
            }

            const fns = Object.values(lock.functions).filter(f => f.moduleId === moduleId)
            const files = Object.values(lock.files).filter(f => f.moduleId === moduleId)

            const detail = {
                module: mod,
                files: files.map(f => ({ path: f.path, imports: f.imports })),
                functions: fns.map(f => ({
                    name: f.name,
                    file: f.file,
                    startLine: f.startLine,
                    endLine: f.endLine,
                    isExported: f.isExported,
                    isAsync: f.isAsync,
                    params: f.params,
                    returnType: f.returnType,
                    calls: f.calls.map(id => lock.functions[id]?.name).filter(Boolean),
                    calledBy: f.calledBy.map(id => lock.functions[id]?.name).filter(Boolean),
                })),
                exported: fns.filter(f => f.isExported).map(f => f.name),
                internal: fns.filter(f => !f.isExported).map(f => f.name),
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(detail, null, 2) }] }
        },
    )


    // TOOL: mikk_get_function_detail

    server.tool(
        'mikk_get_function_detail',
        '360-degree view of a function: params, return type, source body, call graph (who calls it + what it calls), error handling, edge cases. WHEN TO USE: When you need to understand a specific function in depth. AFTER THIS: Use mikk_find_usages to see all callers. TIP: Pass full qualified name (e.g. GraphBuilder.build) for class methods.',
        {
            name: z.string().describe('Function name to search for (e.g., "parseFiles", "GraphBuilder.build")'),
        },
        async ({ name }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)

            const nameLower = name.toLowerCase()
            const matches = Object.values(lock.functions).filter(
                f => f.name.toLowerCase() === nameLower || f.name.toLowerCase().endsWith(`.${nameLower}`) || f.id.toLowerCase().includes(nameLower),
            )

            if (matches.length === 0) {
                return {
                    content: [{ type: 'text' as const, text: `No function matching "${name}" found. Use mikk_search_functions to find the correct name.` }],
                    isError: true,
                }
            }

            const results = await Promise.all(matches.map(async fn => {
                let body: string | undefined
                try {
                    const absPath = path.isAbsolute(fn.file)
                        ? fn.file
                        : path.join(projectRoot, fn.file)
                    const resolved = path.resolve(absPath)
                    const rootResolved = path.resolve(projectRoot)
                    if (!resolved.startsWith(rootResolved + path.sep) && resolved !== rootResolved) {
                        throw new Error('Access denied')
                    }

                    const rel = path.relative(rootResolved, resolved).replace(/\\/g, '/')
                    const allowlisted = new Set(['mikk.json', 'mikk.lock.json', 'package.json', 'tsconfig.json'])
                    if (!(rel in lock.files) && !allowlisted.has(rel)) {
                        throw new Error('Access denied')
                    }

                    const stat = await fs.stat(resolved)
                    if (stat.size > MAX_SOURCE_FILE_BYTES) {
                        throw new Error('File too large')
                    }

                    const fileContent = await fs.readFile(resolved, 'utf-8')
                    const lines = fileContent.split('\n')
                    body = lines.slice(fn.startLine - 1, fn.endLine).join('\n')
                } catch { /* non-fatal — body may not be available */ }

                return {
                    id: fn.id,
                    name: fn.name,
                    file: fn.file,
                    lines: `${fn.startLine}-${fn.endLine}`,
                    module: fn.moduleId,
                    isExported: fn.isExported,
                    isAsync: fn.isAsync,
                    params: fn.params,
                    returnType: fn.returnType,
                    purpose: fn.purpose,
                    body,
                    calls: fn.calls.map(id => lock.functions[id]?.name).filter(Boolean),
                    calledBy: fn.calledBy.map(id => lock.functions[id]?.name).filter(Boolean),
                    errorHandling: fn.errorHandling,
                    edgeCases: fn.edgeCasesHandled,
                    warning: staleness,
                }
            }))

            return { content: [{ type: 'text' as const, text: JSON.stringify(results, null, 2) }] }
        },
    )


    // TOOL: mikk_semantic_search

    server.tool(
        'mikk_semantic_search',
        'Find functions by meaning using local vector embeddings. Query "validate JWT" returns verifyToken ranked by cosine similarity. WHEN TO USE: When you dont know the function name but know what it does. Complements mikk_search_functions (keyword). AFTER THIS: Use mikk_get_function_detail on top matches. Requires @xenova/transformers (22MB model, downloads once).',
        {
            query: z.string().min(1).max(500).describe('Natural-language description of what you are looking for (e.g. "validate a JWT token", "send an email notification")'),
            topK: z.number().int().min(1).max(50).optional().default(10).describe('Number of results to return (default: 10)'),
        },
        async ({ query, topK }) => {
            const available = await SemanticSearcher.isAvailable()
            if (!available) {
                return {
                    content: [{
                        type: 'text' as const,
                        text: [
                            '⚠ Semantic search requires @xenova/transformers.',
                            '',
                            'Install it in your project root:',
                            '  npm install @xenova/transformers',
                            '  # or: pnpm add @xenova/transformers',
                            '',
                            'Tip: mikk_search_functions works right now for exact keyword search.',
                        ].join('\n'),
                    }],
                    isError: true,
                }
            }

            const { lock, staleness } = await loadContractAndLock(projectRoot)
            const searcher = getSemanticSearcher(projectRoot)

            let matches: Awaited<ReturnType<typeof searcher.search>>
            try {
                await searcher.index(lock)
                matches = await searcher.search(query, lock, topK)
            } catch (err: any) {
                return {
                    content: [{
                        type: 'text' as const,
                        text: `Semantic search failed: ${err?.message ?? String(err)}. Try mikk_search_functions as fallback.`,
                    }],
                    isError: true,
                }
            }

            const response = {
                query,
                method: 'semantic (vector similarity)',
                model: SemanticSearcher.MODEL,
                matches,
                tip: 'Use mikk_search_functions for exact substring search instead.',
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_validate_edit (NEW - Uses IntentUnderstanding, AutoCorrection, SafetyGates)
    server.tool(
        'mikk_validate_edit',
        'MANDATORY: Use BEFORE any edit. Combines intent analysis, impact assessment, auto-correction, and enforced safety gates. Tells you if edit is allowed, what breaks, and auto-fixes issues. WHEN TO USE: Always before modifying files. AFTER THIS: If allowed, proceed with edit. If blocked, follow nextSteps.',
        {
            files: z.array(z.string()).min(1).max(20).describe('Files you plan to edit (relative paths)'),
            description: z.string().describe('What are you trying to accomplish?'),
            commitMessage: z.string().optional().describe('Planned commit message (helps detect intent)'),
            branchName: z.string().optional().describe('Current branch name (helps detect intent)'),
            autoFix: z.boolean().optional().default(true).describe('Apply automatic fixes?'),
        },
        async ({ files, description, commitMessage, branchName, autoFix }) => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)
            const graph = buildGraphFromLock(lock)
            
            // Import new intent-engine capabilities
            const { PreEditValidation } = await import('@getmikk/intent-engine')
            
            const validator = new PreEditValidation(contract, lock, graph, projectRoot)
            
            const proposal = {
                files,
                description,
                author: 'AI Assistant',
                intent: {
                    commitMessage,
                    branchName,
                    filesChanged: files,
                    changeType: 'unknown',
                    confidence: 0.7
                }
            }
            
            const result = await validator.validate(proposal)
            
            // Build response
            const response = {
                allowed: result.allowed,
                confidence: result.confidence,
                
                intent: {
                    isIntentionalBreakingChange: result.intent.isIntentionalBreakingChange,
                    confidence: result.intent.confidence,
                    reasoning: result.intent.reasoning,
                    riskAcceptance: result.intent.riskAcceptance
                },
                
                impact: {
                    totalFiles: result.impact.totalFiles,
                    totalFunctions: result.impact.totalFunctions,
                    riskScore: result.impact.riskScore,
                    criticalPaths: result.impact.criticalPaths,
                    blastRadius: result.impact.blastRadius
                },
                
                gates: result.gates.map(g => ({
                    name: g.name,
                    passed: g.passed,
                    severity: g.severity,
                    message: g.message
                })),
                
                corrections: result.corrections,
                
                recommendations: result.recommendations,
                nextSteps: result.nextSteps,
                tokenSavings: result.tokenSavings,
                
                warning: staleness,
                hint: result.allowed 
                    ? '✓ Edit approved. Review recommendations before proceeding.'
                    : '✗ Edit blocked. Address blocking gates first.',
            }
            
            const _rawVal = files.length * Math.round((200 * _ALC) / _CPT)
                ; (response as any).tokens = _track(projectRoot, _rawVal, response)
            
            return { 
                content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }],
                isError: !result.allowed
            }
        },
    )


    // TOOL: mikk_get_constraints

    server.tool(
        'mikk_get_constraints',
        'Get all architectural constraints and ADRs. WHEN TO USE: Before cross-module changes, or when mikk_before_edit reports violations. Understand WHY a constraint exists. AFTER THIS: Use mikk_manage_adr to add/update decisions. 6 constraint types: no-import, must-use, no-call, layer, naming, max-files.',
        {},
        async () => {
            const { contract, staleness } = await loadContractAndLock(projectRoot)

            const result = {
                constraints: contract.declared.constraints,
                decisions: contract.declared.decisions,
                overwrite: contract.overwrite,
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }] }
        },
    )


    // TOOL: mikk_get_file

    server.tool(
        'mikk_get_file',
        'Read raw source of a file. TIP: Prefer mikk_read_file with function names to save tokens. WHEN TO USE: When you need entire file content (config files, small files). AFTER THIS: Use mikk_before_edit before making changes.',
        {
            file: z.string().describe('File path relative to project root (e.g., "src/auth/verify.ts")'),
        },
        async ({ file }) => {
            try {
                const absPath = path.isAbsolute(file) ? file : path.join(projectRoot, file)

                // Guard against path traversal
                const resolved = path.resolve(absPath)
                const rootResolved = path.resolve(projectRoot)
                if (!resolved.startsWith(rootResolved + path.sep) && resolved !== rootResolved) {
                    return {
                        content: [{ type: 'text' as const, text: `Access denied: "${file}" is outside the project root.` }],
                        isError: true,
                    }
                }

                const stat = await fs.stat(resolved)
                if (stat.size > MAX_SOURCE_FILE_BYTES) {
                    return {
                        content: [{ type: 'text' as const, text: `Refusing to read "${file}" because it exceeds ${MAX_SOURCE_FILE_BYTES} bytes.` }],
                        isError: true,
                    }
                }
                const rel = path.relative(path.resolve(projectRoot), resolved).replace(/\\/g, '/')
                const { lock } = await loadContractAndLock(projectRoot)
                const allowlisted = new Set(['mikk.json', 'mikk.lock.json', 'package.json', 'tsconfig.json'])
                const isTracked = rel in lock.files
                if (!isTracked && !allowlisted.has(rel)) {
                    return {
                        content: [{ type: 'text' as const, text: `Access denied: "${file}" is not tracked in mikk.lock.json.` }],
                        isError: true,
                    }
                }
                const content = await fs.readFile(resolved, 'utf-8')
                const lineCount = content.split('\n').length
                return {
                    content: [{
                        type: 'text' as const,
                        text: `// ${file} (${lineCount} lines)\n${content}`,
                    }],
                }
            } catch (err: any) {
                return {
                    content: [{ type: 'text' as const, text: `Cannot read "${file}": ${err.message}. Use mikk_search_functions to find the correct path.` }],
                    isError: true,
                }
            }
        },
    )


    // TOOL: mikk_find_usages

    server.tool(
        'mikk_find_usages',
        'Find every function that calls a specific function. Essential before renaming or changing signatures. WHEN TO USE: Before renaming, refactoring, or changing a function interface. AFTER THIS: Review each caller to ensure your change wont break them. Use mikk_read_file to see caller code.',
        {
            name: z.string().describe('Function name to find callers of'),
        },
        async ({ name }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)

            const nameLower = name.toLowerCase()
            const fn = Object.values(lock.functions).find(
                f => f.name.toLowerCase() === nameLower || f.name.toLowerCase().endsWith(`.${nameLower}`) || f.id.toLowerCase().includes(nameLower),
            )

            if (!fn) {
                return {
                    content: [{ type: 'text' as const, text: `Function "${name}" not found. Use mikk_search_functions to verify the name.` }],
                    isError: true,
                }
            }

            const usages = fn.calledBy
                .map(id => lock.functions[id])
                .filter(Boolean)
                .map(caller => ({
                    name: caller.name,
                    file: caller.file,
                    module: caller.moduleId,
                    line: caller.startLine,
                }))

            const response = {
                function: fn.name,
                file: fn.file,
                module: fn.moduleId,
                usageCount: usages.length,
                usages,
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_get_routes

    server.tool(
        'mikk_get_routes',
        'Get all detected HTTP routes with methods, paths, handlers, and middleware chains. WHEN TO USE: When working on API endpoints. Shows Express/Koa/Hono route registrations detected from AST. AFTER THIS: Use mikk_get_function_detail on a handler to see its implementation.',
        {},
        async () => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)
            const routes = lock.routes ?? []

            if (routes.length === 0) {
                return { content: [{ type: 'text' as const, text: 'No HTTP routes detected in this project.' }] }
            }

            const response = {
                routes,
                warning: staleness,
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_dead_code

    server.tool(
        'mikk_dead_code',
        'Detect dead code — functions with zero callers after exempting exports, entry points, route handlers, tests, and constructors. Use this before refactoring or cleanup.',
        {
            moduleId: z.string().optional().describe('Filter results to a specific module ID'),
        },
        async ({ moduleId }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)
            const graph = buildGraphFromLock(lock)
            const detector = new DeadCodeDetector(graph, lock)
            const result = detector.detect()

            const filtered = moduleId
                ? {
                    ...result,
                    deadFunctions: result.deadFunctions.filter(f => f.moduleId === moduleId),
                    deadCount: result.deadFunctions.filter(f => f.moduleId === moduleId).length,
                    byModule: { [moduleId]: result.byModule[moduleId] ?? { dead: 0, total: 0, items: [] } },
                }
                : result

            const response = {
                ...filtered,
                warning: staleness,
                hint: 'Next: Review dead functions and consider removing them. Use mikk_get_function_detail on any function to see its full context before removing.',
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_manage_adr

    server.tool(
        'mikk_manage_adr',
        'CRUD for Architectural Decision Records (ADRs) in mikk.json. Actions: list, get, add, update, remove. WHEN TO USE: When making architectural changes — document WHY so future AI agents understand. AFTER THIS: ADRs automatically surface in mikk_query_context responses. Required for add: id, title, reason.',
        {
            action: z.enum(['list', 'get', 'add', 'update', 'remove']).describe('The CRUD action to perform'),
            id: z.string().optional().describe('ADR id (required for get, update, remove)'),
            title: z.string().optional().describe('ADR title (required for add)'),
            reason: z.string().optional().describe('ADR reason/description (required for add)'),
            date: z.string().optional().describe('ADR date string (defaults to today for add)'),
        },
        async ({ action, id, title, reason, date }) => {
            const contractPath = path.join(projectRoot, 'mikk.json')
            const manager = new AdrManager(contractPath)

            try {
                switch (action) {
                    case 'list': {
                        const decisions = await manager.list()
                        return {
                            content: [{
                                type: 'text' as const, text: JSON.stringify({
                                    decisions,
                                    count: decisions.length,
                                    hint: 'Next: Use "get" with an ADR id for details, or "add" to create a new decision.',
                                }, null, 2)
                            }],
                        }
                    }
                    case 'get': {
                        if (!id) return { content: [{ type: 'text' as const, text: 'Error: "id" is required for get action.' }], isError: true }
                        const decision = await manager.get(id)
                        if (!decision) return { content: [{ type: 'text' as const, text: `ADR "${id}" not found.` }], isError: true }
                        return { content: [{ type: 'text' as const, text: JSON.stringify(decision, null, 2) }] }
                    }
                    case 'add': {
                        if (!id || !title || !reason) {
                            return { content: [{ type: 'text' as const, text: 'Error: "id", "title", and "reason" are required for add action.' }], isError: true }
                        }
                        await manager.add({ id, title, reason, date: date ?? new Date().toISOString().split('T')[0] })
                        return { content: [{ type: 'text' as const, text: `ADR "${id}" added to mikk.json. This decision will now surface in all AI context queries.` }] }
                    }
                    case 'update': {
                        if (!id) return { content: [{ type: 'text' as const, text: 'Error: "id" is required for update action.' }], isError: true }
                        await manager.update(id, { ...(title ? { title } : {}), ...(reason ? { reason } : {}), ...(date ? { date } : {}) })
                        return { content: [{ type: 'text' as const, text: `ADR "${id}" updated.` }] }
                    }
                    case 'remove': {
                        if (!id) return { content: [{ type: 'text' as const, text: 'Error: "id" is required for remove action.' }], isError: true }
                        const removed = await manager.remove(id)
                        return { content: [{ type: 'text' as const, text: removed ? `ADR "${id}" removed.` : `ADR "${id}" not found.` }] }
                    }
                }
            } catch (err: any) {
                return { content: [{ type: 'text' as const, text: `ADR operation failed: ${err.message}` }], isError: true }
            }
        },
    )


    // TOOL: mikk_get_changes  (Phase 2)

    server.tool(
        'mikk_get_changes',
        'Detect files added, modified, and deleted since last mikk analyze. WHEN TO USE: At session start (after mikk_get_session_context), or after making edits to see what drifted. AFTER THIS: Run mikk analyze to update the lock, then mikk_impact_analysis on modified files. Uses SHA-256 hash comparison for accurate drift detection.',
        {},
        async () => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)

            const added: string[] = []
            const modified: string[] = []
            const deleted: string[] = []
            let scanTruncated = false

            for (const [filePath, fileInfo] of Object.entries(lock.files)) {
                const absPath = path.isAbsolute(filePath)
                    ? filePath
                    : path.join(projectRoot, filePath)

                try {
                    const currentHash = await quickHashFile(absPath)
                    const storedHash = fileInfo.hash?.slice(0, 16) ?? ''
                    if (currentHash !== storedHash && storedHash !== '') {
                        modified.push(filePath)
                    }
                } catch {
                    deleted.push(filePath)
                }
            }

            // Check for new files not in the lock
            try {
                const srcDirs = ['src', 'lib', 'app', 'pages', 'components']
                for (const dir of srcDirs) {
                    const dirPath = path.join(projectRoot, dir)
                    try {
                        await fs.access(dirPath)
                        const files = await walkDir(dirPath, projectRoot)
                        if (files.length >= MAX_WALK_FILES) scanTruncated = true
                        for (const f of files) {
                            if (!lock.files[f] && isSourceFile(f)) {
                                added.push(f)
                            }
                        }
                    } catch { /* dir doesn't exist */ }
                }
            } catch { /* scan failed — non-fatal */ }

            const response = {
                added: added.slice(0, 50),
                modified: modified.slice(0, 50),
                deleted: deleted.slice(0, 50),
                summary: `${modified.length} modified, ${added.length} new, ${deleted.length} deleted since last analysis`,
                totalChanges: added.length + modified.length + deleted.length,
                warning: staleness,
                hint: modified.length + added.length > 0
                    ? 'Run `mikk analyze` to update the lock file with these changes.'
                    : 'Codebase is in sync with the lock file.',
            }

            if (scanTruncated) {
                response.hint += `\nNote: change scan was truncated after ${MAX_WALK_FILES} files for performance.`
            }

            // Token savings: replaces grep/find across repo for changed files + hashing manually
            const _rawGC = Math.min(50, Object.keys(lock.files).length) * Math.round((60 * _ALC) / _CPT)
                ; (response as any).tokens = _track(projectRoot, _rawGC, response)
            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )


    // TOOL: mikk_read_file  (Phase 2)

    server.tool(
        'mikk_read_file',
        'Read file scoped to specific functions. Returns bodies with metadata headers (params, calls, calledBy). WHEN TO USE: When you know which functions you need — saves tokens vs mikk_get_file. AFTER THIS: Use mikk_before_edit before making changes. TIP: This is the preferred way to read code — always specify function names when possible.',
        {
            file: z.string().describe('File path relative to project root'),
            functions: z.array(z.string()).max(30).optional().describe('Function names to extract. If omitted, returns the whole file.'),
        },
        async ({ file, functions: fnNames }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)

            const absPath = path.isAbsolute(file) ? file : path.join(projectRoot, file)
            const resolved = path.resolve(absPath)
            const rootResolved = path.resolve(projectRoot)
            if (!resolved.startsWith(rootResolved + path.sep) && resolved !== rootResolved) {
                return {
                    content: [{ type: 'text' as const, text: `Access denied: "${file}" is outside the project root.` }],
                    isError: true,
                }
            }

            let content: string
            try {
                const stat = await fs.stat(resolved)
                if (stat.size > MAX_SOURCE_FILE_BYTES) {
                    return {
                        content: [{ type: 'text' as const, text: `Refusing to read "${file}" because it exceeds ${MAX_SOURCE_FILE_BYTES} bytes.` }],
                        isError: true,
                    }
                }
                const rel = path.relative(path.resolve(projectRoot), resolved).replace(/\\/g, '/')
                const allowlisted = new Set(['mikk.json', 'mikk.lock.json', 'package.json', 'tsconfig.json'])
                const isTracked = rel in lock.files
                if (!isTracked && !allowlisted.has(rel)) {
                    return {
                        content: [{ type: 'text' as const, text: `Access denied: "${file}" is not tracked in mikk.lock.json.` }],
                        isError: true,
                    }
                }
                content = await fs.readFile(resolved, 'utf-8')
            } catch (err: any) {
                return {
                    content: [{ type: 'text' as const, text: `Cannot read "${file}": ${err.message}` }],
                    isError: true,
                }
            }

            if (!fnNames || fnNames.length === 0) {
                const lines = content.split('\n')
                return {
                    content: [{ type: 'text' as const, text: `// ${file} (${lines.length} lines)\n${content}` }],
                }
            }

            const lines = content.split('\n')
            const sections: string[] = []
            const normalizedFile = file.replace(/\\/g, '/')

            for (const fnName of fnNames) {
                const fnNameLower = fnName.toLowerCase()
                const fn = Object.values(lock.functions).find(
                    f => (f.name.toLowerCase() === fnNameLower || f.name.toLowerCase().endsWith(`.${fnNameLower}`)) &&
                        (f.file === normalizedFile || f.file.endsWith('/' + normalizedFile))
                )

                if (!fn) {
                    sections.push(`// ⚠ Function "${fnName}" not found in ${file}`)
                    continue
                }

                const header = [
                    `//  ${fn.name} `,
                    `// File: ${fn.file}:${fn.startLine}-${fn.endLine}`,
                    `// Module: ${fn.moduleId}`,
                    fn.purpose ? `// Purpose: ${fn.purpose}` : null,
                    fn.params && fn.params.length > 0 ? `// Params: ${fn.params.map(p => `${p.name}: ${p.type}`).join(', ')}` : null,
                    fn.returnType ? `// Returns: ${fn.returnType}` : null,
                    fn.isAsync ? '// Async: true' : null,
                    fn.isExported ? '// Exported: true' : null,
                    fn.calledBy.length > 0 ? `// Called by: ${fn.calledBy.map(id => lock.functions[id]?.name).filter(Boolean).join(', ')}` : null,
                    fn.calls.length > 0 ? `// Calls: ${fn.calls.map(id => lock.functions[id]?.name).filter(Boolean).join(', ')}` : null,
                ].filter(Boolean).join('\n')

                const body = lines.slice(fn.startLine - 1, fn.endLine).join('\n')
                sections.push(`${header}\n${body}`)
            }

            const output = sections.join('\n\n')
            const warningText = staleness ? `\n\n${staleness}` : ''

            // Token savings: reading specific functions saves tokens vs whole-file read
            const _rawRF = _fileTok(lock, file.replace(/\\/g, '/'))
            const _tokRF = _track(projectRoot, _rawRF, output)
            return { content: [{ type: 'text' as const, text: output + warningText + `\n// tokens: ${JSON.stringify(_tokRF)}` }] }
        },
    )

    // TOOL: mikk_get_session_context  (Phase 2)
    server.tool(
        'mikk_get_session_context',
        'CALL THIS FIRST. One-shot context for session start: project overview + constraint status + hot modules + recently modified files + active decisions. WHEN TO USE: At the very beginning of every AI conversation. This is your onboarding. AFTER THIS: Use mikk_query_context with your task description, or mikk_get_changes for detailed drift.',
        {},
        async () => {
            const { contract, lock, staleness } = await loadContractAndLock(projectRoot)

            const modules = contract.declared.modules.map(mod => {
                const fns = Object.values(lock.functions).filter(f => f.moduleId === mod.id)
                return {
                    id: mod.id,
                    name: mod.name,
                    functions: fns.length,
                    exported: fns.filter(f => f.isExported).length,
                }
            })

            // Detect recent changes via mtime comparison
            let changedCount = 0
            const modifiedFiles: string[] = []
            const fileEntries = Object.entries(lock.files)
            const sampleSize = Math.min(fileEntries.length, 20)
            for (let i = 0; i < sampleSize; i++) {
                const [filePath, fileInfo] = fileEntries[i]
                const absPath = path.isAbsolute(filePath) ? filePath : path.join(projectRoot, filePath)
                try {
                    const stat = await fs.stat(absPath)
                    const lockDate = new Date(fileInfo.lastModified || 0)
                    if (stat.mtime > lockDate) {
                        modifiedFiles.push(filePath)
                        changedCount++
                    }
                } catch { changedCount++ }
            }


            const moduleChanges = new Map<string, number>()
            for (const f of modifiedFiles) {
                const fileInfo = lock.files[f]
                if (fileInfo?.moduleId) {
                    moduleChanges.set(fileInfo.moduleId, (moduleChanges.get(fileInfo.moduleId) ?? 0) + 1)
                }
            }
            const hotModules = [...moduleChanges.entries()]
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([id, changes]) => ({ id, changes }))

            // Constraint status check
            const checker = new BoundaryChecker(contract, lock)
            const boundaryResult = checker.check()

            const response = {
                project: contract.project,
                summary: {
                    totalFunctions: Object.keys(lock.functions).length,
                    totalFiles: Object.keys(lock.files).length,
                    totalModules: modules.length,
                    constraintViolations: boundaryResult.violations.length,
                    constraintsPass: boundaryResult.pass,
                    estimatedChanges: changedCount,
                },
                modules,
                hotModules,
                recentlyModified: modifiedFiles.slice(0, 10),
                constraints: contract.declared.constraints,
                decisions: contract.declared.decisions.slice(0, 5),
                warning: staleness,
                hint: changedCount > 0
                    ? `${changedCount} file(s) may have changed. Run \`mikk analyze\` for accurate results, or use mikk_get_changes for details.`
                    : 'Codebase is in sync. Use mikk_query_context with your task description to get started.',
            }

            // Token savings: session_context replaces reading all module files individually
            const _rawSC = Math.min(20, Object.keys(lock.files).length) * Math.round((100 * _ALC) / _CPT)
                ; (response as any).tokens = _track(projectRoot, _rawSC, response)
            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )

    // TOOL: mikk_git_diff_impact
    server.tool(
        'mikk_git_diff_impact',
        'Map git diff hunks to affected symbols. Shows which functions were modified/added/deleted. WHEN TO USE: After commits/merges to understand symbol-level changes. AFTER THIS: Use mikk_impact_analysis on affected files.',
        {
            ref: z.string().optional().default('HEAD~1').describe('Git ref to diff against (default: HEAD~1)'),
            staged: z.boolean().optional().default(false).describe('If true, diff staged changes only'),
        },
        async ({ ref, staged }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)
            try {
                const validatedRef = /^[A-Za-z0-9_./\-~^]+$/.test(ref) ? ref : null
                if (!staged && !validatedRef) {
                    return {
                        content: [{ type: 'text' as const, text: 'Invalid git ref format.' }],
                        isError: true,
                    }
                }
                const args = ['diff']
                if (staged) args.push('--cached')
                else args.push(validatedRef!)
                args.push('--unified=0', '--no-color')
                const rawDiff = await new Promise<string>((resolve, reject) => {
                    execFile('git', args, { cwd: projectRoot, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }, (err, stdout) => {
                        if (err) return reject(err)
                        resolve(stdout)
                    })
                })
                if (!rawDiff.trim()) {
                    return { content: [{ type: 'text' as const, text: 'No changes found in git diff.' }] }
                }
                const fileHunks = parseDiffHunks(rawDiff)
                const affectedSymbols: { file: string; type: string; functions: { name: string; moduleId: string }[] }[] = []
                for (const hunk of fileHunks) {
                    const fileFns = Object.values(lock.functions).filter(fn => fn.file === hunk.file || fn.file.endsWith(hunk.file))
                    const affected = fileFns.filter(fn => hunk.changedLines.some(l => l >= fn.startLine && l <= fn.endLine))
                    if (affected.length > 0 || hunk.isNew || hunk.isDeleted) {
                        affectedSymbols.push({
                            file: hunk.file,
                            type: hunk.isNew ? 'added' : hunk.isDeleted ? 'deleted' : 'modified',
                            functions: affected.map(fn => ({ name: fn.name, moduleId: fn.moduleId })),
                        })
                    }
                }
                const totalFns = affectedSymbols.reduce((s, f) => s + f.functions.length, 0)
                return {
                    content: [{
                        type: 'text' as const, text: JSON.stringify({
                            summary: `${affectedSymbols.length} file(s), ${totalFns} function(s) affected`,
                            affectedSymbols, warning: staleness,
                        }, null, 2)
                    }]
                }
            } catch (err: any) {
                return { content: [{ type: 'text' as const, text: `Git diff failed: ${err.message}` }], isError: true }
            }
        },
    )

    // TOOL: mikk_rename
    server.tool(
        'mikk_rename',
        'Plan a coordinated multi-file rename. Finds all call sites and import locations for a function and provides a step-by-step edit plan. WHEN TO USE: Before renaming any function — ensures you update ALL call sites. AFTER THIS: Execute the edit plan, then run mikk analyze.',
        {
            functionName: z.string().describe('The current function name to rename'),
            newName: z.string().describe('The desired new name'),
        },
        async ({ functionName, newName }) => {
            const { lock, staleness } = await loadContractAndLock(projectRoot)

            const targetFn = Object.values(lock.functions).find(fn =>
                fn.name === functionName || fn.id.endsWith(`:${functionName}`)
            )

            if (!targetFn) {
                return {
                    content: [{
                        type: 'text' as const,
                        text: `Function "${functionName}" not found. Use mikk_search_functions to find the correct name.`,
                    }],
                    isError: true,
                }
            }

            const callers = targetFn.calledBy
                .map(callerId => lock.functions[callerId])
                .filter(Boolean)
                .map(fn => ({
                    callerName: fn.name,
                    file: fn.file,
                    module: fn.moduleId,
                    lineRange: `${fn.startLine}-${fn.endLine}`,
                }))

            const filesImporting = Object.values(lock.files).filter(file =>
                file.imports?.some(imp => imp.names.includes(functionName) || imp.source === targetFn.file)
            )

            const instructions = [
                `1. Rename definition in ${targetFn.file}:${targetFn.startLine}`,
                ...callers.map((c, i) => `${i + 2}. Update call in ${c.file} (${c.callerName}, lines ${c.lineRange})`),
                ...(targetFn.isExported
                    ? filesImporting.map((f, i) => `${callers.length + i + 2}. Update import in ${f.path}`)
                    : []),
                `${callers.length + (targetFn.isExported ? filesImporting.length : 0) + 2}. Run \`mikk analyze\` to update the lock`,
            ]

            return {
                content: [{
                    type: 'text' as const,
                    text: JSON.stringify({
                        target: {
                            currentName: functionName,
                            newName,
                            file: targetFn.file,
                            line: targetFn.startLine,
                            module: targetFn.moduleId,
                            isExported: targetFn.isExported,
                        },
                        callSites: callers,
                        importSites: filesImporting.map(f => ({ file: f.path, module: f.moduleId })),
                        totalEdits: 1 + callers.length + filesImporting.length,
                        instructions,
                        warning: staleness,
                    }, null, 2),
                }],
            }
        },
    )
    // TOOL: mikk_token_stats
    server.tool(
        'mikk_token_stats',
        'Show token savings for this session — how many tokens Mikk saved vs. the agent reading raw source files. WHEN TO USE: Any time. Useful at end of session to see cumulative efficiency. Returns per-session totals and cost estimates.',
        {},
        async () => {
            const t = _tally(projectRoot)
            const { lock } = await loadContractAndLock(projectRoot)
            const totalFileLine = Object.values(lock.functions).reduce((s, f) => s + (f.endLine - f.startLine + 1), 0)
            const fullCodebaseTok = Math.round((totalFileLine * _ALC) / _CPT)
            const elapsedMin = Math.round((Date.now() - t.start) / 60000)

            const response = {
                session: {
                    calls: t.calls,
                    elapsedMinutes: elapsedMin,
                },
                tokens: {
                    used: t.used,
                    rawWouldHaveCost: t.raw,
                    saved: t.saved,
                    savingsPercent: t.raw > 0 ? Math.round((t.saved / t.raw) * 100) : 0,
                },
                context: {
                    fullCodebaseTokens: fullCodebaseTok,
                    percentOfCodebaseRead: t.raw > 0 ? Math.round((t.used / fullCodebaseTok) * 100) : 0,
                    note: 'Full codebase = if agent read every tracked source line once',
                },
                interpretation: t.saved > 0
                    ? `Mikk saved ~${t.saved.toLocaleString()} tokens this session (${Math.round((t.saved / t.raw) * 100)}% reduction). Roughly ${Math.round(t.saved / 1000)}k tokens = ~${(t.saved * 0.000003).toFixed(3)} USD at GPT-4o rates.`
                    : 'No tools called yet this session.',
            }

            return { content: [{ type: 'text' as const, text: JSON.stringify(response, null, 2) }] }
        },
    )
}

/**
 * Load contract + lock from disk with 30s caching and active staleness detection.
 * Cache is invalidated immediately when the lock file's mtime is newer than
 * cachedAt — this means `mikk analyze` takes effect on the very next tool call,
 * not after a 30s wait.
 */
async function loadContractAndLock(projectRoot: string) {
    // Check lock file mtime first — if the file changed since we cached, bust immediately.
    const lockFilePath = path.join(projectRoot, 'mikk.lock.json')
    const cached = projectCache.get(projectRoot)
    if (cached) {
        try {
            const stat = await fs.stat(lockFilePath)
            if (stat.mtimeMs > cached.cachedAt) {
                // Lock file was written after we cached (e.g. `mikk analyze` ran) — invalidate
                invalidateCache(projectRoot)
            } else if ((Date.now() - cached.cachedAt) < CACHE_TTL_MS) {
                // Still within TTL and lock file unchanged — serve from cache
                return { contract: cached.contract, lock: cached.lock, staleness: cached.staleness }
            }
        } catch {
            // stat failed (lock deleted?) — fall through to re-read
            invalidateCache(projectRoot)
        }
    }

    const contractReader = new ContractReader()
    const lockReader = new LockReader()
    const contract = await contractReader.read(path.join(projectRoot, 'mikk.json'))
    const lock = await lockReader.read(path.join(projectRoot, 'mikk.lock.json'))

    // Self-reported staleness
    const syncStatus = lock.syncState?.status ?? 'unknown'
    let staleness: string | null = null

    if (syncStatus === 'drifted' || syncStatus === 'conflict') {
        staleness = `⚠ Lock file is ${syncStatus}. Run \`mikk analyze\` for accurate results.`
    }

    // Active staleness detection: check mtime of a sample of tracked files
    if (!staleness) {
        const fileEntries = Object.entries(lock.files)
        const sampleSize = Math.min(fileEntries.length, 5)
        let mismatched = 0
        const mismatchedFiles: string[] = []

        for (let i = 0; i < sampleSize; i++) {
            const [filePath, fileInfo] = fileEntries[i]
            const absPath = path.isAbsolute(filePath)
                ? filePath
                : path.join(projectRoot, filePath)

            try {
                const stat = await fs.stat(absPath)
                const lockDate = new Date(fileInfo.lastModified || 0)
                if (stat.mtime > lockDate) {
                    mismatched++
                    mismatchedFiles.push(filePath)
                }
            } catch {
                mismatched++ // file deleted
                mismatchedFiles.push(filePath)
            }
        }

        if (mismatched > 0) {
            staleness = `⚠ STALE: ${mismatched} file(s) changed since last analysis (${mismatchedFiles.slice(0, 3).join(', ')}${mismatched > 3 ? '...' : ''}). Run \`mikk analyze\`.`
        }
    }

    // Build graph and cache everything
    const graph = buildGraphFromLock(lock)
    projectCache.set(projectRoot, {
        contract, lock, graph, staleness,
        cachedAt: Date.now(),
    })

    return { contract, lock, staleness }
}

/**
 * Build a DependencyGraph from the lock file in O(n) time.
 * The lock already has fn.calls and fn.calledBy arrays we just wire them up.
 */
function buildGraphFromLock(lock: MikkLock): DependencyGraph {
    const nodes = new Map<string, GraphNode>()
    const edges: GraphEdge[] = []
    const outEdges = new Map<string, GraphEdge[]>()
    const inEdges = new Map<string, GraphEdge[]>()

    for (const fn of Object.values(lock.functions)) {
        nodes.set(fn.id, {
            id: fn.id,
            type: 'function',
            name: fn.name,
            file: fn.file,
            moduleId: fn.moduleId,
            metadata: {
                startLine: fn.startLine,
                endLine: fn.endLine,
                isExported: fn.isExported,
                isAsync: fn.isAsync,
                hash: fn.hash,
                purpose: fn.purpose,
                params: fn.params,
                returnType: fn.returnType,
                edgeCasesHandled: fn.edgeCasesHandled,
                errorHandling: fn.errorHandling,
            },
        })
    }

    for (const file of Object.values(lock.files)) {
        nodes.set(file.path, {
            id: file.path,
            type: 'file',
            name: path.basename(file.path),
            file: file.path,
            moduleId: file.moduleId,
            metadata: {},
        })
    }

    for (const fn of Object.values(lock.functions)) {
        for (const calleeId of fn.calls) {
            if (!nodes.has(calleeId)) continue
            const edge: GraphEdge = { from: fn.id, to: calleeId, type: 'calls', confidence: 1.0 }
            edges.push(edge)

            const out = outEdges.get(fn.id) ?? []
            out.push(edge)
            outEdges.set(fn.id, out)

            const inE = inEdges.get(calleeId) ?? []
            inE.push(edge)
            inEdges.set(calleeId, inE)
        }
    }

    return { nodes, edges, outEdges, inEdges }
}

/** Detect circular dependencies for a set of functions via DFS */
function detectCircularDeps(
    fns: MikkLock['functions'][string][],
    lock: MikkLock
): string[] {
    const warnings: string[] = []

    for (const fn of fns) {
        const visited = new Set<string>()
        const stack = new Set<string>()
        const cyclePath: string[] = []

        function dfs(id: string): boolean {
            if (stack.has(id)) {
                const cycleStart = cyclePath.indexOf(id)
                const cycle = cyclePath.slice(cycleStart).map(cid => lock.functions[cid]?.name ?? cid)
                cycle.push(lock.functions[id]?.name ?? id)
                warnings.push(`⚠ Circular: ${cycle.join(' -> ')}`)
                return true
            }
            if (visited.has(id)) return false

            visited.add(id)
            stack.add(id)
            cyclePath.push(id)

            const callee = lock.functions[id]
            if (callee) {
                for (const callId of callee.calls) {
                    if (dfs(callId)) return true
                }
            }

            stack.delete(id)
            cyclePath.pop()
            return false
        }

        dfs(fn.id)
    }

    return [...new Set(warnings)]
}

/** Recursively walk a directory and return relative file paths (bounded for safety). */
async function walkDir(
    dir: string,
    projectRoot: string,
    depth = 0,
    acc: string[] = [],
): Promise<string[]> {
    if (depth > MAX_WALK_DIR_DEPTH || acc.length >= MAX_WALK_FILES) return acc

    try {
        const entries = await fs.readdir(dir, { withFileTypes: true })
        for (const entry of entries) {
            if (acc.length >= MAX_WALK_FILES) break

            const fullPath = path.join(dir, entry.name)
            if (entry.isSymbolicLink()) continue

            if (entry.isDirectory()) {
                if (entry.name === 'node_modules' || entry.name === '.git' || entry.name === 'dist' || entry.name === '.mikk') continue
                await walkDir(fullPath, projectRoot, depth + 1, acc)
            } else {
                acc.push(path.relative(projectRoot, fullPath).replace(/\\/g, '/'))
            }
        }
    } catch { /* permission error or similar */ }

    return acc
}

/** Check if a file is a source file worth tracking */
function isSourceFile(filePath: string): boolean {
    const ext = path.extname(filePath)
    return ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.go', '.py'].includes(ext)
}

/** Parse unified diff into per-file hunk info with changed line numbers */
function parseDiffHunks(diff: string): { file: string; changedLines: number[]; isNew: boolean; isDeleted: boolean }[] {
    const files = new Map<string, { changedLines: number[]; isNew: boolean; isDeleted: boolean }>()
    let currentFile = ''
    let nextIsNew = false

    for (const line of diff.split('\n')) {
        if (line.startsWith('--- ') && line.includes('/dev/null')) {
            nextIsNew = true
        } else if (line.startsWith('+++ ')) {
            currentFile = line.slice(6)
            if (currentFile !== '/dev/null' && !files.has(currentFile)) {
                files.set(currentFile, { changedLines: [], isNew: nextIsNew, isDeleted: false })
            }
            if (currentFile === '/dev/null') {
                // deletion — mark previous file
                const prev = [...files.keys()].pop()
                if (prev) files.get(prev)!.isDeleted = true
            }
            nextIsNew = false
        } else if (line.startsWith('@@ ') && currentFile && files.has(currentFile)) {
            const match = line.match(/@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/)
            if (match) {
                const start = parseInt(match[1], 10)
                const count = parseInt(match[2] ?? '1', 10)
                const entry = files.get(currentFile)!
                for (let i = 0; i < count; i++) entry.changedLines.push(start + i)
            }
        }
    }

    return [...files.entries()].map(([file, data]) => ({ file, ...data }))
}
