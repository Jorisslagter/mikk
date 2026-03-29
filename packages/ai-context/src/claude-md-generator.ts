import type { MikkContract, MikkLock, MikkLockFunction } from '@getmikk/core'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { countTokens, estimateFileTokens } from './token-counter.js'

/** Default token budget for claude.md — generous but still bounded */
const DEFAULT_TOKEN_BUDGET = 12000

/** Metadata from package.json that enriches the AI context */
export interface ProjectMeta {
    description?: string
    scripts?: Record<string, string>
    dependencies?: Record<string, string>
    devDependencies?: Record<string, string>
}

/**
 * ClaudeMdGenerator — generates an always-accurate `claude.md` and `AGENTS.md`
 * from the lock file and contract. Every function name, file path, and module
 * relationship is sourced from the AST-derived lock file — never hand-authored.
 *
 * Tiered system per spec:
 *  Tier 1: Summary (~500 tokens) — always included
 *  Tier 2: Module details (~300 tokens/module) — included if budget allows
 *  Tier 3: Recent changes (~50 tokens/change) — last section added
 */
export class ClaudeMdGenerator {
    private meta: ProjectMeta

    constructor(
        private contract: MikkContract,
        private lock: MikkLock,
        private tokenBudget: number = DEFAULT_TOKEN_BUDGET,
        meta?: ProjectMeta,
        private projectRoot?: string
    ) {
        this.meta = meta || {}
    }

    /** Generate the full claude.md content */
    generate(): string {
        const sections: string[] = []
        let usedTokens = 0

        // --- Tier 1: Summary (always included) ----------------------
        const summary = this.generateSummary()
        sections.push(summary)
        usedTokens += countTokens(summary)

        // --- MCP tool instructions (always included) ---
        const mikkInstructions = this.generateMikkInstructions()
        sections.push(mikkInstructions)
        usedTokens += countTokens(mikkInstructions)

        // --- Tech stack & conventions (always included if detectable) ---
        const techSection = this.generateTechStackSection()
        if (techSection) {
            sections.push(techSection)
            usedTokens += countTokens(techSection)
        }

        // --- Build / test / run commands -----------------------------
        const commandsSection = this.generateCommandsSection()
        if (commandsSection) {
            sections.push(commandsSection)
            usedTokens += countTokens(commandsSection)
        }

        // --- Tier 2: Module details (if budget allows) --------------
        // Skip modules with zero functions — they waste AI tokens
        const modules = this.getModulesSortedByDependencyOrder()
            .filter(m => {
                const fnCount = Object.values(this.lock.functions)
                    .filter(f => f.moduleId === m.id).length
                return fnCount > 0
            })

        for (const module of modules) {
            const moduleSection = this.generateModuleSection(module.id)
            const tokens = countTokens(moduleSection)
            if (usedTokens + tokens > this.tokenBudget) {
                sections.push('\n  <!-- Full details truncated due to context budget -->\n')
                break
            }
            sections.push(moduleSection)
            usedTokens += tokens
        }

        sections.push('</modules>\n')

        // --- Context files: schemas, data models, config ---------
        const contextSection = this.generateContextFilesSection()
        if (contextSection) {
            const ctxTokens = countTokens(contextSection)
            if (usedTokens + ctxTokens <= this.tokenBudget) {
                sections.push(contextSection)
                usedTokens += ctxTokens
            }
        }

        // --- File import graph per module ----------------------------
        const importSection = this.generateImportGraphSection()
        if (importSection) {
            const impTokens = countTokens(importSection)
            if (usedTokens + impTokens <= this.tokenBudget) {
                sections.push(importSection)
                usedTokens += impTokens
            }
        }

        // --- HTTP Routes (Express + Next.js) -------------------------
        const routesSection = this.generateRoutesSection()
        if (routesSection) {
            const routeTokens = countTokens(routesSection)
            if (usedTokens + routeTokens <= this.tokenBudget) {
                sections.push(routesSection)
                usedTokens += routeTokens
            }
        }

        // --- Tier 3: Constraints & decisions ------------------------
        const constraintsSection = this.generateConstraintsSection()
        const constraintTokens = countTokens(constraintsSection)
        if (usedTokens + constraintTokens <= this.tokenBudget) {
            sections.push(constraintsSection)
            usedTokens += constraintTokens
        }

        const decisionsSection = this.generateDecisionsSection()
        const decisionTokens = countTokens(decisionsSection)
        if (usedTokens + decisionTokens <= this.tokenBudget) {
            sections.push(decisionsSection)
            usedTokens += decisionTokens
        }

        return sections.join('\n')
    }

    // ── MCP Tool Instructions ──────────────────────────────────────

    private generateMikkInstructions(): string {
        return `
## Mikk MCP Tools — MANDATORY

This project uses Mikk MCP tools for code exploration. You MUST prefer these over raw Explore agents, Grep, Glob, or Read when understanding code.

**Before exploring code:** Call \`mikk_query_context\` with your question — it traces the dependency graph and returns relevant function bodies.
**To find functions:** Call \`mikk_search_functions\` (by name) or \`mikk_semantic_search\` (by meaning).
**To read a function:** Call \`mikk_get_function_detail\` — returns params, calls, calledBy, body, and error handling.
**Before editing any file:** Call \`mikk_before_edit\` — checks blast radius and constraints.
**After edits:** Call \`mikk_get_changes\` to verify drift.
`
    }

    // ── Tier 1: Summary ───────────────────────────────────────────

    private generateSummary(): string {
        const lines: string[] = []
        const moduleCount = this.contract.declared.modules.length
        const functionCount = Object.keys(this.lock.functions).length
        const fileCount = Object.keys(this.lock.files).length

        lines.push('<repository_context>')
        lines.push(`  <name>${this.contract.project.name}</name>`)

        // Project description: prefer contract, fall back to package.json
        const description = this.contract.project.description || this.meta.description
        if (description) {
            lines.push(`  <description>${description}</description>`)
        }

        // Only list modules that have functions (skip empty ones)
        const nonEmptyModules = this.contract.declared.modules.filter(m => {
            const fnCount = Object.values(this.lock.functions)
                .filter(f => f.moduleId === m.id).length
            return fnCount > 0
        })

        lines.push(`  <stats>`)
        lines.push(`    <files>${fileCount}</files>`)
        lines.push(`    <functions>${functionCount}</functions>`)
        lines.push(`    <modules>${nonEmptyModules.length}</modules>`)
        lines.push(`    <language>${this.contract.project.language}</language>`)
        lines.push(`  </stats>`)

        // Critical constraints summary
        if (this.contract.declared.constraints.length > 0) {
            lines.push('  <critical_constraints>')
            for (const c of this.contract.declared.constraints) {
                lines.push(`    <constraint>${c}</constraint>`)
            }
            lines.push('  </critical_constraints>')
        }

        lines.push('</repository_context>')
        lines.push('')
        lines.push('<modules>')

        return lines.join('\n')
    }

    // ── Tier 2: Module Details ────────────────────────────────────

    private generateModuleSection(moduleId: string): string {
        const module = this.contract.declared.modules.find(m => m.id === moduleId)
        if (!module) return ''

        const lines: string[] = []
        const moduleFunctions = Object.values(this.lock.functions)
            .filter(f => f.moduleId === moduleId)

        lines.push(`  <module id="${moduleId}">`)
        lines.push(`    <name>${module.name}</name>`)

        // Location — collapse to common prefix when many paths share a root
        if (module.paths.length > 0) {
            const collapsed = this.collapsePaths(module.paths)
            lines.push(`    <location>${collapsed}</location>`)
        }

        // Intent
        if (module.intent) {
            lines.push(`    <purpose>${module.intent}</purpose>`)
        } else if (module.description) {
            lines.push(`    <purpose>${module.description}</purpose>`)
        }

        // Entry points: functions with no calledBy (likely public API surface)
        const entryPoints = moduleFunctions
            .filter(fn => fn.calledBy.length === 0)
            .sort((a, b) => b.calls.length - a.calls.length)
            .slice(0, 5)

        if (entryPoints.length > 0) {
            lines.push('    <entry_points>')
            for (const fn of entryPoints) {
                const sig = this.formatSignature(fn)
                const purpose = fn.purpose ? `${this.oneLine(fn.purpose)}` : ''
                lines.push(`      <function signature="${sig.replace(/"/g, '&quot;')}" purpose="${purpose.replace(/"/g, '&quot;')}" />`)
            }
            lines.push('    </entry_points>')
        }

        // Key functions: top 5 by calledBy count (most depended upon)
        // Exclude functions already in entry points to avoid duplicates
        const entryPointIds = new Set(entryPoints.map(fn => fn.id))
        const keyFunctions = [...moduleFunctions]
            .filter(fn => !entryPointIds.has(fn.id)) // Exclude duplicates
            .sort((a, b) => b.calledBy.length - a.calledBy.length)
            .filter(fn => fn.calledBy.length > 0)
            .slice(0, 5)

        if (keyFunctions.length > 0) {
            lines.push('    <key_internal_functions>')
            for (const fn of keyFunctions) {
                const callerCount = fn.calledBy.length
                const purpose = fn.purpose ? `${this.oneLine(fn.purpose)}` : ''
                lines.push(`      <function name="${fn.name.replace(/"/g, '&quot;')}" callers="${callerCount}" purpose="${purpose.replace(/"/g, '&quot;')}" />`)
            }
            lines.push('    </key_internal_functions>')
        }

        // Dependencies: other modules this module imports from
        const depModuleIds = new Set<string>()
        for (const fn of moduleFunctions) {
            for (const callId of fn.calls) {
                const target = this.lock.functions[callId]
                if (target && target.moduleId !== moduleId) {
                    depModuleIds.add(target.moduleId)
                }
            }
        }

        if (depModuleIds.size > 0) {
            const depNames = [...depModuleIds].map(id => {
                const mod = this.contract.declared.modules.find(m => m.id === id)
                return mod?.name || id
            })
            lines.push(`    <depends_on>${depNames.join(', ')}</depends_on>`)
        }

        // Module-specific constraints
        const moduleConstraints = this.contract.declared.constraints.filter(c =>
            c.toLowerCase().includes(moduleId.toLowerCase()) ||
            c.toLowerCase().includes(module.name.toLowerCase())
        )
        if (moduleConstraints.length > 0) {
            lines.push('    <module_constraints>')
            for (const c of moduleConstraints) {
                lines.push(`      <constraint>${c}</constraint>`)
            }
            lines.push('    </module_constraints>')
        }

        lines.push(`  </module>`)
        return lines.join('\n')
    }

    // ── Tier 3: Constraints & Decisions ───────────────────────────

    private generateConstraintsSection(): string {
        if (this.contract.declared.constraints.length === 0) return ''
        const lines: string[] = []
        lines.push('## Cross-Cutting Constraints')
        for (const c of this.contract.declared.constraints) {
            lines.push(`- ${c}`)
        }
        lines.push('')
        return lines.join('\n')
    }

    private generateDecisionsSection(): string {
        if (this.contract.declared.decisions.length === 0) return ''
        const lines: string[] = []
        lines.push('## Architectural Decisions')
        for (const d of this.contract.declared.decisions) {
            lines.push(`- **${d.title}:** ${d.reason}`)
        }
        lines.push('')
        return lines.join('\n')
    }

    // ── Context Files Section ──────────────────────────────────────

    /** Generate a section with discovered schema/config files inlined */
    private generateContextFilesSection(): string | null {
        const ctxFiles = this.lock.contextFiles
        if (!ctxFiles || ctxFiles.length === 0) return null

        const lines: string[] = []
        lines.push('## Data Models & Schemas')
        lines.push('')
        lines.push('These files define the project\'s data structures, schemas, and configuration.')
        lines.push('They are auto-discovered and included verbatim from the source.')
        lines.push('')

        for (const cf of ctxFiles) {
            const ext = cf.path.split('.').pop() || 'txt'
            const lang = this.extToLang(ext)
            lines.push(`### \`${cf.path}\` (${cf.type})`)
            lines.push('')
            lines.push('```' + lang)
            // Read content from disk on-demand
            let content = ''
            if (this.projectRoot) {
                try { content = fs.readFileSync(path.resolve(this.projectRoot, cf.path), 'utf-8') } catch { }
            }
            // Trim content to avoid blowing up the token budget
            const maxChars = 8000 // ~2000 tokens per file
            if (content.length > maxChars) {
                lines.push(content.slice(0, maxChars))
                lines.push(`// ... truncated (${cf.size ?? content.length} bytes total)`)
            } else {
                lines.push(content.trimEnd())
            }
            lines.push('```')
            lines.push('')
        }

        return lines.join('\n')
    }

    /** Map file extensions to Markdown code fence languages */
    private extToLang(ext: string): string {
        const map: Record<string, string> = {
            ts: 'typescript', js: 'javascript', tsx: 'tsx', jsx: 'jsx',
            mjs: 'javascript', cjs: 'javascript',
            py: 'python', go: 'go', rs: 'rust', rb: 'ruby',
            java: 'java', kt: 'kotlin', cs: 'csharp', swift: 'swift',
            dart: 'dart', ex: 'elixir', exs: 'elixir', php: 'php',
            prisma: 'prisma', graphql: 'graphql', gql: 'graphql',
            sql: 'sql', proto: 'protobuf', yaml: 'yaml', yml: 'yaml',
            json: 'json', toml: 'toml', xml: 'xml',
            css: 'css', scss: 'scss', html: 'html',
            svelte: 'svelte', vue: 'vue',
            md: 'markdown', sh: 'bash', bash: 'bash', zsh: 'bash',
            dockerfile: 'dockerfile', tf: 'hcl',
            avsc: 'json', thrift: 'thrift',
        }
        return map[ext.toLowerCase()] || ext
    }

    // ── Routes Section ──────────────────────────────────────────

    /** Generate a section with detected HTTP route registrations + Next.js filesystem routes */
    private generateRoutesSection(): string | null {
        const expressRoutes = this.lock.routes || []
        const nextRoutes = this.detectNextJsRoutes()
        if (expressRoutes.length === 0 && nextRoutes.length === 0) return null

        const lines: string[] = []
        lines.push('## HTTP Routes')
        lines.push('')

        // Next.js App Router routes (detected from file paths)
        if (nextRoutes.length > 0) {
            lines.push('### API Routes (Next.js App Router)')
            for (const r of nextRoutes) {
                const methods = r.methods.length > 0 ? r.methods.join(', ') : 'handler'
                lines.push(`- **${methods}** \`${r.urlPath}\` *(${r.file})*`)
            }
            lines.push('')
        }

        // Express/Koa/Hono routes
        if (expressRoutes.length > 0) {
            if (nextRoutes.length > 0) lines.push('### Server Routes')
            for (const r of expressRoutes) {
                const mw = r.middlewares.length > 0 ? ` → [${r.middlewares.join(', ')}]` : ''
                lines.push(`- **${r.method}** \`${r.path}\` → \`${r.handler}\`${mw} *(${r.file}:${r.line})*`)
            }
            lines.push('')
        }

        return lines.join('\n')
    }

    // ── Tech Stack Section ──────────────────────────────────────

    /** Detect technology stack from dependencies and config */
    private generateTechStackSection(): string | null {
        const deps = { ...this.meta.dependencies, ...this.meta.devDependencies }
        if (Object.keys(deps).length === 0) return null

        const detected: string[] = []

        // Language / Runtime
        const lang = this.contract.project.language
        if (lang && lang !== 'typescript' && lang !== 'javascript') {
            detected.push(lang.charAt(0).toUpperCase() + lang.slice(1))
        }

        // Frameworks — JS/TS
        if (deps['next']) detected.push(`Next.js ${deps['next'].replace(/^\^|~/, '')}`)
        else if (deps['nuxt']) detected.push(`Nuxt ${deps['nuxt'].replace(/^\^|~/, '')}`)
        else if (deps['@sveltejs/kit']) detected.push('SvelteKit')
        else if (deps['@remix-run/react'] || deps['@remix-run/node']) detected.push('Remix')
        else if (deps['astro']) detected.push('Astro')
        else if (deps['gatsby']) detected.push('Gatsby')
        else if (deps['express']) detected.push('Express')
        else if (deps['fastify']) detected.push('Fastify')
        else if (deps['hono']) detected.push('Hono')
        else if (deps['koa']) detected.push('Koa')
        else if (deps['nestjs'] || deps['@nestjs/core']) detected.push('NestJS')
        if (deps['react']) detected.push('React')
        if (deps['vue']) detected.push('Vue')
        if (deps['svelte']) detected.push('Svelte')
        if (deps['solid-js']) detected.push('SolidJS')
        if (deps['angular'] || deps['@angular/core']) detected.push('Angular')

        // Mobile / Desktop
        if (deps['react-native']) detected.push('React Native')
        if (deps['expo']) detected.push('Expo')
        if (deps['@capacitor/core']) detected.push('Capacitor')
        if (deps['electron']) detected.push('Electron')
        if (deps['tauri'] || deps['@tauri-apps/api']) detected.push('Tauri')

        // Database / ORM
        if (deps['prisma'] || deps['@prisma/client']) detected.push('Prisma ORM')
        else if (deps['drizzle-orm']) detected.push('Drizzle ORM')
        else if (deps['typeorm']) detected.push('TypeORM')
        else if (deps['mongoose']) detected.push('Mongoose')
        else if (deps['sequelize']) detected.push('Sequelize')
        else if (deps['knex']) detected.push('Knex')
        else if (deps['pg'] || deps['mysql2'] || deps['better-sqlite3']) detected.push('SQL client')
        if (deps['redis'] || deps['ioredis']) detected.push('Redis')

        // Auth
        if (deps['next-auth'] || deps['@auth/core']) detected.push('NextAuth')
        else if (deps['passport']) detected.push('Passport.js')
        else if (deps['clerk'] || deps['@clerk/nextjs']) detected.push('Clerk')
        else if (deps['lucia'] || deps['lucia-auth']) detected.push('Lucia')

        // Styling
        if (deps['tailwindcss']) detected.push('Tailwind CSS')
        if (deps['radix-ui'] || deps['@radix-ui/react-dialog'] || deps['@radix-ui/react-slot']) detected.push('Radix UI')
        if (deps['shadcn'] || deps['shadcn-ui']) detected.push('shadcn/ui')
        if (deps['styled-components']) detected.push('styled-components')
        if (deps['@emotion/react']) detected.push('Emotion')

        // State / Data
        if (deps['@tanstack/react-query']) detected.push('TanStack Query')
        if (deps['zustand']) detected.push('Zustand')
        if (deps['jotai']) detected.push('Jotai')
        if (deps['@reduxjs/toolkit'] || deps['redux']) detected.push('Redux')
        if (deps['zod']) detected.push('Zod validation')
        if (deps['@trpc/server'] || deps['@trpc/client']) detected.push('tRPC')
        if (deps['graphql'] || deps['@apollo/client']) detected.push('GraphQL')

        // Content / Docs
        if (deps['fumadocs-core'] || deps['fumadocs-mdx']) detected.push('Fumadocs')
        if (deps['next-mdx-remote'] || deps['@next/mdx']) detected.push('MDX')
        if (deps['contentlayer'] || deps['contentlayer2']) detected.push('Contentlayer')

        // Animation
        if (deps['motion'] || deps['framer-motion']) detected.push('Motion')

        // Analytics / Monitoring
        if (deps['posthog-js'] || deps['@posthog/react']) detected.push('PostHog')
        if (deps['@vercel/analytics']) detected.push('Vercel Analytics')
        if (deps['@sentry/node'] || deps['@sentry/nextjs'] || deps['@sentry/browser']) detected.push('Sentry')

        // URL state
        if (deps['nuqs']) detected.push('nuqs')

        // Messaging / Queue
        if (deps['bullmq'] || deps['bull']) detected.push('BullMQ')
        if (deps['amqplib']) detected.push('RabbitMQ')
        if (deps['kafkajs']) detected.push('Kafka')

        // Cloud / Infra
        if (deps['aws-sdk'] || deps['@aws-sdk/client-s3']) detected.push('AWS SDK')
        if (deps['@google-cloud/storage']) detected.push('Google Cloud')
        if (deps['firebase'] || deps['firebase-admin']) detected.push('Firebase')
        if (deps['@supabase/supabase-js']) detected.push('Supabase')
        if (deps['convex']) detected.push('Convex')

        // Testing
        if (deps['jest']) detected.push('Jest')
        else if (deps['vitest']) detected.push('Vitest')
        if (deps['playwright'] || deps['@playwright/test']) detected.push('Playwright')
        if (deps['cypress']) detected.push('Cypress')
        if (deps['@testing-library/react']) detected.push('Testing Library')

        // Build
        if (deps['turbo'] || deps['turbo-json']) detected.push('Turborepo')
        if (deps['nx']) detected.push('Nx')
        if (deps['webpack']) detected.push('Webpack')
        if (deps['vite']) detected.push('Vite')
        if (deps['esbuild']) detected.push('esbuild')
        if (deps['rollup']) detected.push('Rollup')

        if (detected.length === 0) return null

        const lines: string[] = []
        lines.push('<tech_stack>')
        for (const d of detected) {
            lines.push(`  <technology>${d}</technology>`)
        }
        lines.push('</tech_stack>')
        return lines.join('\n')
    }

    // ── Commands Section ────────────────────────────────────────

    /** Detect build/test/dev commands from package.json scripts */
    private generateCommandsSection(): string | null {
        const scripts = this.meta.scripts
        if (!scripts || Object.keys(scripts).length === 0) return null

        // Auto-detect package manager from lockfile hints in meta, or fall back to heuristic
        let pm = 'npm run'
        const deps = { ...this.meta.dependencies, ...this.meta.devDependencies }
        // Heuristic: check for telltale signs in scripts values
        const allScriptValues = Object.values(scripts).join(' ')
        if (allScriptValues.includes('bun ') || deps['bun-types']) pm = 'bun run'
        else if (allScriptValues.includes('pnpm ') || allScriptValues.includes('pnpm-')) pm = 'pnpm'
        else if (allScriptValues.includes('yarn ')) pm = 'yarn'

        const useful: [string, string][] = []
        const interestingKeys = ['dev', 'build', 'start', 'test', 'lint', 'format', 'typecheck', 'check', 'e2e', 'storybook', 'db:push', 'db:migrate', 'db:seed', 'prisma:generate', 'generate']

        for (const key of interestingKeys) {
            if (scripts[key]) {
                useful.push([key, scripts[key]])
            }
        }

        if (useful.length === 0) return null

        const lines: string[] = []
        lines.push('<commands>')
        for (const [key, cmd] of useful) {
            lines.push(`  <command>`)
            lines.push(`    <run>${pm} ${key}</run>`)
            lines.push(`    <executes>${cmd.replace(/"/g, '&quot;')}</executes>`)
            lines.push(`  </command>`)
        }
        lines.push('</commands>')
        return lines.join('\n')
    }

    // ── Next.js Route Detection ─────────────────────────────────

    /** Detect Next.js App Router routes from file paths in the lock */
    private detectNextJsRoutes(): { urlPath: string; methods: string[]; file: string }[] {
        const routes: { urlPath: string; methods: string[]; file: string }[] = []
        const allFiles = Object.keys(this.lock.files)

        for (const filePath of allFiles) {
            const normalised = filePath.replace(/\\/g, '/')

            // App Router API routes: app/**/route.ts
            const routeMatch = normalised.match(/^(?:src\/)?app\/(.+)\/route\.[jt]sx?$/)
            if (routeMatch) {
                const urlSegments = routeMatch[1]
                    .replace(/\([\w-]+\)\/?/g, '')           // strip route groups (marketing)/
                    .replace(/\[\.\.\.(\w+)\]/g, ':$1*')     // [...slug] → :slug*
                    .replace(/\[(\w+)\]/g, ':$1')            // [id] → :id
                const urlPath = `/${urlSegments}` || '/'

                // Detect exported HTTP methods from functions AND generics
                // (Next.js handlers can be `export async function GET` → fn: or `export const GET =` → const:)
                const methods: string[] = []
                const HTTP_VERBS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
                for (const [fnId, fn] of Object.entries(this.lock.functions)) {
                    if (fn.file === filePath && fn.isExported && HTTP_VERBS.includes(fn.name)) {
                        methods.push(fn.name)
                    }
                }
                if (this.lock.generics) {
                    for (const [gId, g] of Object.entries(this.lock.generics)) {
                        // Direct match: this generic is in this file
                        if (g.file === filePath && g.isExported && HTTP_VERBS.includes(g.name)) {
                            if (!methods.includes(g.name)) methods.push(g.name)
                        }
                        // alsoIn match: deduped generics list the same verb in other files
                        if (g.isExported && HTTP_VERBS.includes(g.name) && g.alsoIn?.includes(filePath)) {
                            if (!methods.includes(g.name)) methods.push(g.name)
                        }
                    }
                }

                routes.push({ urlPath, methods, file: filePath })
            }

            // App Router pages: app/**/page.tsx (informational)
            const pageMatch = normalised.match(/^(?:src\/)?app\/(.+)\/page\.[jt]sx?$/)
            if (pageMatch) {
                const urlSegments = pageMatch[1]
                    .replace(/\([\w-]+\)\/?/g, '')           // strip route groups
                    .replace(/\[\.\.\.(\w+)\]/g, ':$1*')
                    .replace(/\[(\w+)\]/g, ':$1')
                const urlPath = `/${urlSegments}` || '/'
                routes.push({ urlPath, methods: ['PAGE'], file: filePath })
            }
        }

        // Sort by URL path
        return routes.sort((a, b) => a.urlPath.localeCompare(b.urlPath))
    }

    // ── Import Graph Section ──────────────────────────────────────

    /** Generate a per-module file import map */
    private generateImportGraphSection(): string | null {
        const filesWithImports = Object.values(this.lock.files)
            .filter(f => f.imports && f.imports.length > 0)
        if (filesWithImports.length === 0) return null

        const lines: string[] = []
        lines.push('## File Import Graph')
        lines.push('')
        lines.push('Which files import which — useful for understanding data flow.')
        lines.push('')

        // Group by module
        const byModule = new Map<string, typeof filesWithImports>()
        for (const f of filesWithImports) {
            const existing = byModule.get(f.moduleId) || []
            existing.push(f)
            byModule.set(f.moduleId, existing)
        }

        for (const [moduleId, files] of byModule) {
            const mod = this.contract.declared.modules.find(m => m.id === moduleId)
            const name = mod?.name || moduleId
            lines.push(`### ${name}`)
            for (const f of files) {
                const imports = f.imports!.map(imp => `\`${typeof imp === 'string' ? imp : imp.source}\``).join(', ')
                lines.push(`- \`${f.path}\` → ${imports}`)
            }
            lines.push('')
        }

        return lines.join('\n')
    }

    // ── Helpers ───────────────────────────────────────────────────

    /** Collapse multi-line text to a single trimmed line */
    private oneLine(text: string): string {
        return text.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim()
    }

    /** Format a function into a compact single-line signature */
    private formatSignature(fn: MikkLockFunction): string {
        const asyncPrefix = fn.isAsync ? 'async ' : ''
        const params = fn.params && fn.params.length > 0
            ? fn.params.map(p => {
                const opt = p.optional ? '?' : ''
                // Use just name (skip destructured object types — they bloat multi-line)
                const name = p.name.replace(/\n/g, ' ').replace(/\s+/g, ' ')
                return `${name}${opt}`
            }).join(', ')
            : ''
        return `${asyncPrefix}${fn.name}(${params}) [${fn.file}:${fn.startLine}]`
    }

    /**
     * Collapse many paths into fewest glob patterns.
     * e.g. ["src/features/portfolio/components/awards/**", "src/features/portfolio/components/bookmarks/**", ...]
     * becomes "src/features/portfolio/**"
     */
    private collapsePaths(paths: string[]): string {
        if (paths.length <= 2) return paths.join(', ')

        // Split each path into segments (strip trailing **)
        const stripped = paths.map(p => p.replace(/\/\*\*$/, ''))

        // Try progressively shorter common prefixes
        // Find the longest common directory prefix shared by majority of paths
        const segments = stripped.map(p => p.split('/'))
        let bestPrefix = ''
        for (let depth = 1; depth <= (segments[0]?.length ?? 0); depth++) {
            const prefix = segments[0].slice(0, depth).join('/')
            const matching = stripped.filter(p => p === prefix || p.startsWith(prefix + '/'))
            if (matching.length >= Math.ceil(paths.length * 0.6)) {
                bestPrefix = prefix
            }
        }

        if (bestPrefix) {
            // Some paths outside the prefix
            const outside = paths.filter(p => {
                const s = p.replace(/\/\*\*$/, '')
                return s !== bestPrefix && !s.startsWith(bestPrefix + '/')
            })
            if (outside.length === 0) {
                return `${bestPrefix}/**`
            }
            return [`${bestPrefix}/**`, ...outside].join(', ')
        }

        return paths.join(', ')
    }

    /** Sort modules by inter-module dependency order (depended-on modules first) */
    private getModulesSortedByDependencyOrder(): typeof this.contract.declared.modules {
        const modules = [...this.contract.declared.modules]
        const dependencyCount = new Map<string, number>()

        for (const mod of modules) {
            dependencyCount.set(mod.id, 0)
        }

        // Count how many other modules depend on each module
        for (const fn of Object.values(this.lock.functions)) {
            for (const callId of fn.calls) {
                const target = this.lock.functions[callId]
                if (target && target.moduleId !== fn.moduleId) {
                    dependencyCount.set(
                        target.moduleId,
                        (dependencyCount.get(target.moduleId) || 0) + 1
                    )
                }
            }
        }

        // Sort: most depended-on first
        return modules.sort((a, b) =>
            (dependencyCount.get(b.id) || 0) - (dependencyCount.get(a.id) || 0)
        )
    }
}
