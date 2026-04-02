import path from 'node:path';
import { BaseParser } from './base-parser.js';
import { OxcResolver } from './oxc-resolver.js';
import { hashContent } from '../hash/file-hasher.js';
import type {
    ParsedFile,
    ParsedFunction,
    ParsedClass,
    ParsedVariable,
    ParsedImport,
    ParsedExport,
    ParsedParam,
    CallExpression,
    ParsedGeneric,
    ParsedRoute
} from './types.js';

// ---------------------------------------------------------------------------
// LineIndex — O(log n) byte-offset → 1-based line number
// ---------------------------------------------------------------------------
class LineIndex {
    private readonly offsets: number[];

    constructor(content: string) {
        this.offsets = [0];
        let i = 0;
        while ((i = content.indexOf('\n', i)) !== -1) {
            this.offsets.push(++i);
        }
    }

    getLine(offset: number): number {
        let lo = 0;
        let hi = this.offsets.length - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >>> 1;
            if (this.offsets[mid] <= offset) lo = mid + 1;
            else hi = mid - 1;
        }
        return hi + 1; // 1-based
    }
}

// ---------------------------------------------------------------------------
// ID allocation
// ---------------------------------------------------------------------------
// Canonical ID format (lowercased for stable matching):
//   fn:<absolute-posix-path>:<functionname>
//   fn:<absolute-posix-path>:<functionname>#2   (second occurrence in same file)
//   class:<absolute-posix-path>:<classname>
//   type:<absolute-posix-path>:<typename>
//   enum:<absolute-posix-path>:<enumname>
//   var:<absolute-posix-path>:<varname>
//   prop:<absolute-posix-path>:<propname>
// ---------------------------------------------------------------------------
function makeAllocator(filePath: string): (prefix: string, name: string) => string {
    const counter = new Map<string, number>();
    const normalizedPath = filePath.replace(/\\/g, '/');
    return (prefix: string, name: string): string => {
        const key = `${prefix}:${name}`;
        const count = (counter.get(key) ?? 0) + 1;
        counter.set(key, count);
        const suffix = count === 1 ? '' : `#${count}`;
        return `${prefix}:${normalizedPath}:${name}${suffix}`.toLowerCase();
    };
}

// ---------------------------------------------------------------------------
// Export detection helpers
// ---------------------------------------------------------------------------
function isDirectlyExported(parent: any): boolean {
    return parent != null && (
        parent.type === 'ExportNamedDeclaration' ||
        parent.type === 'ExportDeclaration'
    );
}

// ---------------------------------------------------------------------------
// AST helpers
// ---------------------------------------------------------------------------
const memberExpressionTypes = new Set([
    'MemberExpression',
    'StaticMemberExpression',
    'ComputedMemberExpression',
    'OptionalMemberExpression',
]);

function normalizeCallee(node: any): any {
    if (!node || typeof node !== 'object') return null;
    if (node.type === 'ChainExpression') return normalizeCallee(node.expression);
    if (node.type === 'OptionalCallExpression') return normalizeCallee(node.callee);
    return node;
}

function resolvePropertyName(node: any): string | null {
    if (!node) return null;
    if (node.type === 'Identifier') return node.name ?? null;
    if (node.type === 'PrivateIdentifier') return `#${node.name}`;
    if (node.type === 'Literal' || node.type === 'StringLiteral' || node.type === 'NumericLiteral') {
        return node.value != null ? String(node.value) : node.raw ?? null;
    }
    return null;
}

function resolveObjectName(node: any): string | null {
    if (!node) return null;
    if (node.type === 'Identifier') return node.name;
    if (node.type === 'ThisExpression') return 'this';
    if (node.type === 'Super') return 'super';
    if (memberExpressionTypes.has(node.type)) {
        const parent = resolveObjectName(node.object ?? node.expression);
        const prop = resolvePropertyName(node.property ?? node.expression);
        if (parent && prop) return `${parent}.${prop}`;
        return prop;
    }
    if (node.type === 'NewExpression' || node.type === 'CallExpression') {
        return resolveObjectName(node.callee);
    }
    if (node.type === 'ChainExpression' || node.type === 'OptionalCallExpression') {
        return resolveObjectName(node.expression ?? node.callee);
    }
    return null;
}

function resolveCallIdentity(callee: any): { name: string | null; type: CallExpression['type'] } {
    const normalized = normalizeCallee(callee);
    if (!normalized) return { name: null, type: 'function' };
    if (normalized.type === 'Identifier') {
        return { name: normalized.name ?? null, type: 'function' };
    }
    if (memberExpressionTypes.has(normalized.type)) {
        const objName = resolveObjectName(normalized.object ?? normalized.expression);
        const propName = resolvePropertyName(normalized.property ?? normalized.expression);
        if (objName && propName) {
            return { name: `${objName}.${propName}`, type: 'method' };
        }
        if (propName) {
            return { name: propName, type: 'method' };
        }
        return { name: null, type: 'function' };
    }
    if (normalized.type === 'Super') {
        return { name: 'super', type: 'method' };
    }
    if (normalized.type === 'CallExpression' || normalized.type === 'NewExpression') {
        return resolveCallIdentity(normalized.callee);
    }
    return { name: null, type: 'function' };
}

function flattenPatternNames(pattern: any): string[] {
    if (!pattern) return [];
    switch (pattern.type) {
        case 'Identifier':
            return pattern.name ? [pattern.name] : ['unknown'];
        case 'PrivateIdentifier':
            return pattern.name ? [`#${pattern.name}`] : ['unknown'];
        case 'AssignmentPattern':
            return flattenPatternNames(pattern.left ?? pattern.argument ?? pattern.parameter);
        case 'RestElement':
            return flattenPatternNames(pattern.argument ?? pattern.value);
        case 'TSParameterProperty':
            return flattenPatternNames(pattern.parameter);
        case 'ObjectPattern':
            return (pattern.properties ?? []).flatMap((prop: any) => {
                if (!prop) return [];
                if (prop.type === 'RestElement') return flattenPatternNames(prop.argument);
                return flattenPatternNames(prop.value ?? prop.key);
            });
        case 'ArrayPattern':
            return (pattern.elements ?? []).flatMap((el: any) => flattenPatternNames(el));
        case 'Property':
            return flattenPatternNames(pattern.value ?? pattern.key);
        default:
            return ['unknown'];
    }
}

function normalizeParamNode(node: any): any {
    if (!node) return null;
    if (node.type === 'TSParameterProperty') return normalizeParamNode(node.parameter);
    return node;
}

function describeParamPattern(pattern: any): string {
    if (!pattern) return 'unknown';
    switch (pattern.type) {
        case 'Identifier':
            return pattern.name ?? 'unknown';
        case 'PrivateIdentifier':
            return pattern.name ? `#${pattern.name}` : 'unknown';
        case 'AssignmentPattern':
            return describeParamPattern(pattern.left ?? pattern.argument ?? pattern.parameter);
        case 'RestElement':
            return `...${describeParamPattern(pattern.argument ?? pattern.value ?? pattern.parameter)}`;
        case 'ObjectPattern':
            return '{...}';
        case 'ArrayPattern':
            return '[...]';
        default:
            return 'unknown';
    }
}

function extractTypeParameterNames(typeParameters: any): string[] {
    const params = typeParameters?.params ?? typeParameters?.parameters ?? [];
    if (!Array.isArray(params)) return [];
    return params.map((param: any) => param?.name?.name ?? 'unknown');
}

// ---------------------------------------------------------------------------
// Call extraction
// Captures direct calls (foo()) and method calls (obj.method(), this.method())
// Returns name === 'unknown' only when genuinely unresolvable; those are filtered.
// ---------------------------------------------------------------------------
function extractCalls(node: any, lineIndex: LineIndex): CallExpression[] {
    const calls: CallExpression[] = [];

    const walk = (n: any): void => {
        if (!n || typeof n !== 'object') return;

        if (n.type === 'CallExpression' && n.span) {
            const { name, type } = resolveCallIdentity(n.callee);
            if (name) {
                calls.push({
                    name,
                    line: lineIndex.getLine(n.span.start),
                    type,
                });
            }
        }

        for (const key of Object.keys(n)) {
            if (key === 'span' || key === 'type') continue;
            const child = n[key];
            if (Array.isArray(child)) {
                for (const c of child) walk(c);
            } else if (child && typeof child === 'object') {
                walk(child);
            }
        }
    };

    walk(node);
    return calls;
}

// ---------------------------------------------------------------------------
// Parameter extraction
// ---------------------------------------------------------------------------
function extractParams(params: any[]): ParsedParam[] {
    return params.map(p => {
        const normalized = normalizeParamNode(p);
        const pattern = normalized?.pattern ?? normalized?.left ?? normalized?.argument ?? normalized;
        const name = describeParamPattern(pattern);
        const optional = !!normalized?.optional || pattern?.type === 'AssignmentPattern' || pattern?.type === 'RestElement';
        const hasDefault = pattern?.type === 'AssignmentPattern' || normalized?.defaultValue != null || normalized?.initializer != null;
        return {
            name,
            type: 'any',
            optional,
            defaultValue: hasDefault ? 'default' : undefined,
        };
    });
}

// ---------------------------------------------------------------------------
// Span helper
// ---------------------------------------------------------------------------
function getSpan(node: any): { start: number; end: number } {
    const s = node?.span ?? node ?? {};
    return { start: s.start ?? 0, end: s.end ?? 0 };
}

// ---------------------------------------------------------------------------
// OxcParser
// ---------------------------------------------------------------------------
export class OxcParser extends BaseParser {
    public async parse(filePath: string, content: string): Promise<ParsedFile> {
        const ext = path.extname(filePath).toLowerCase();
        const isTS = ['.ts', '.tsx', '.mts', '.cts'].includes(ext);

        let ast: any;
        try {
            const { parseSync } = await import('oxc-parser');
            const result = parseSync(filePath, content, {
                sourceType: 'module',
                lang: isTS ? 'ts' : 'js',
            });
            ast = result.program;
        } catch {
            // Return empty file on parse error — never crash the pipeline
            return this.emptyFile(filePath, content, isTS);
        }

        const lineIndex = new LineIndex(content);
        const allocateId = makeAllocator(filePath);
        const normalizedFilePath = filePath.replace(/\\/g, '/');

        const functions: ParsedFunction[] = [];
        const classes: ParsedClass[] = [];
        const variables: ParsedVariable[] = [];
        const generics: ParsedGeneric[] = [];
        const imports: ParsedImport[] = [];
        const exports: ParsedExport[] = [];
        const moduleCalls: CallExpression[] = [];
        const routes: ParsedRoute[] = [];

        const visit = (node: any, parent: any = null): void => {
            if (!node || typeof node !== 'object') return;

            switch (node.type) {

                // ── Imports ────────────────────────────────────────────────
                case 'ImportDeclaration': {
                    if (node.importKind === 'type') break;
                    const names: string[] = [];
                    let isDefault = false;
                    for (const spec of node.specifiers ?? []) {
                        if (spec.importKind === 'type') continue;
                        if (spec.type === 'ImportDefaultSpecifier') {
                            isDefault = true;
                        }
                        if (spec.local?.name) names.push(spec.local.name);
                    }
                    imports.push({
                        source: node.source.value,
                        resolvedPath: '',
                        names,
                        isDefault,
                        isDynamic: false,
                    });
                    break;
                }

                // ── Dynamic import() ───────────────────────────────────────
                case 'ImportExpression': {
                    const arg = node.source ?? node.arguments?.[0];
                    if (arg?.type === 'StringLiteral' || arg?.type === 'Literal') {
                        imports.push({
                            source: arg.value,
                            resolvedPath: '',
                            names: [],
                            isDefault: false,
                            isDynamic: true,
                        });
                    }
                    break;
                }

                // ── Function Declaration ───────────────────────────────────
                case 'FunctionDeclaration': {
                    if (!node.id) break;
                    const name = node.id.name;
                    const span = getSpan(node);
                    const exported = isDirectlyExported(parent);
                    functions.push({
                        id: allocateId('fn', name),
                        name,
                        file: normalizedFilePath,
                        startLine: lineIndex.getLine(span.start),
                        endLine: lineIndex.getLine(span.end),
                        params: extractParams(node.params?.items ?? node.params ?? []),
                        returnType: 'void',
                        isExported: exported,
                        isAsync: !!node.async,
                        calls: extractCalls(node.body ?? node, lineIndex),
                        hash: hashContent(JSON.stringify(node.body ?? {})),
                        purpose: '',
                        edgeCasesHandled: [],
                        errorHandling: [],
                        detailedLines: [],
                    });
                    if (exported) exports.push({ name, type: 'function', file: normalizedFilePath });
                    break;
                }

                // ── Class Declaration ──────────────────────────────────────
                case 'ClassDeclaration': {
                    if (!node.id) break;
                    const name = node.id.name;
                    const span = getSpan(node);
                    const exported = isDirectlyExported(parent);
                    const methods: ParsedFunction[] = [];
                    const properties: ParsedVariable[] = [];

                    for (const member of node.body?.body ?? []) {
                        if (member.type === 'MethodDefinition' || member.type === 'PropertyDefinition') {
                            const key = member.key;
                            if (!key) continue;
                            const mName = key.type === 'Identifier' ? key.name :
                                          key.type === 'PrivateIdentifier' ? `#${key.name}` :
                                          null;
                            if (!mName) continue;

                            if (member.type === 'MethodDefinition') {
                                const value = member.value;
                                const mSpan = getSpan(member);
                                methods.push({
                                    id: allocateId('fn', `${name}.${mName}`),
                                    name: `${name}.${mName}`,
                                    file: normalizedFilePath,
                                    startLine: lineIndex.getLine(mSpan.start),
                                    endLine: lineIndex.getLine(mSpan.end),
                                    params: extractParams(value?.params?.items ?? value?.params ?? []),
                                    returnType: 'any',
                                    isExported: exported,
                                    isAsync: !!value?.async,
                                    calls: extractCalls(value?.body ?? value ?? {}, lineIndex),
                                    hash: hashContent(JSON.stringify(value?.body ?? {})),
                                    purpose: '',
                                    edgeCasesHandled: [],
                                    errorHandling: [],
                                    detailedLines: [],
                                });
                            } else {
                                // PropertyDefinition
                                const pSpan = getSpan(member);
                                const propertyNode: ParsedVariable = {
                                    id: allocateId('prop', `${name}.${mName}`),
                                    name: `${name}.${mName}`,
                                    type: 'any',
                                    file: normalizedFilePath,
                                    line: lineIndex.getLine(pSpan.start),
                                    isExported: false,
                                    isStatic: !!member.static,
                                };
                                properties.push(propertyNode);
                                variables.push(propertyNode);
                            }
                        }
                    }

                    classes.push({
                        id: allocateId('class', name),
                        name,
                        file: normalizedFilePath,
                        startLine: lineIndex.getLine(span.start),
                        endLine: lineIndex.getLine(span.end),
                        methods,
                        properties,
                        extends: node.superClass?.name,
                        isExported: exported,
                        hash: hashContent(JSON.stringify(node.body ?? {})),
                        purpose: '',
                    });
                    if (exported) exports.push({ name, type: 'class', file: normalizedFilePath });
                    break;
                }

                // ── TS Type / Interface ────────────────────────────────────
                case 'TSTypeAliasDeclaration':
                case 'TSInterfaceDeclaration': {
                    if (!node.id) break;
                    const name = node.id.name;
                    const span = getSpan(node);
                    const kind = node.type === 'TSInterfaceDeclaration' ? 'interface' : 'type';
                    const exported = isDirectlyExported(parent);
                    const typeParameters = extractTypeParameterNames(node.typeParameters);
                    generics.push({
                        id: allocateId('type', name),
                        name,
                        type: kind,
                        file: normalizedFilePath,
                        startLine: lineIndex.getLine(span.start),
                        endLine: lineIndex.getLine(span.end),
                        isExported: exported,
                        typeParameters,
                        hash: hashContent(JSON.stringify(node)),
                        purpose: '',
                    });
                    if (exported) exports.push({ name, type: kind as any, file: normalizedFilePath });
                    break;
                }

                // ── TS Enum ────────────────────────────────────────────────
                case 'TSEnumDeclaration': {
                    if (!node.id) break;
                    const name = node.id.name;
                    const span = getSpan(node);
                    const exported = isDirectlyExported(parent);
                    generics.push({
                        id: allocateId('enum', name),
                        name,
                        type: 'enum',
                        file: normalizedFilePath,
                        startLine: lineIndex.getLine(span.start),
                        endLine: lineIndex.getLine(span.end),
                        isExported: exported,
                        hash: hashContent(JSON.stringify(node)),
                        purpose: '',
                    });
                    if (exported) exports.push({ name, type: 'const', file: normalizedFilePath });
                    break;
                }

                // ── Variable Declaration ───────────────────────────────────
                case 'VariableDeclaration': {
                    const exported = isDirectlyExported(parent);
                    for (const decl of node.declarations ?? []) {
                        const variableNames = flattenPatternNames(decl.id);
                        if (variableNames.length === 0) continue;

                        // Unwrap TS expressions to find the real initializer
                        let init = decl.init;
                        while (init && (
                            init.type === 'TSAsExpression' ||
                            init.type === 'TSSatisfiesExpression' ||
                            init.type === 'ParenthesizedExpression' ||
                            init.type === 'TypeAssertion' ||
                            init.type === 'TSNonNullExpression' ||
                            init.type === 'TSInstantiationExpression'
                        )) {
                            init = init.expression;
                        }

                        const isFn = init && (
                            init.type === 'FunctionExpression' ||
                            init.type === 'ArrowFunctionExpression'
                        );

                        if (isFn && variableNames.length === 1) {
                            const name = variableNames[0];
                            const span = getSpan(init) ?? getSpan(decl);
                            functions.push({
                                id: allocateId('fn', name),
                                name,
                                file: normalizedFilePath,
                                startLine: lineIndex.getLine(span.start),
                                endLine: lineIndex.getLine(span.end),
                                params: extractParams(init.params?.items ?? init.params ?? []),
                                returnType: 'any',
                                isExported: exported,
                                isAsync: !!init.async,
                                calls: extractCalls(init.body ?? init, lineIndex),
                                hash: hashContent(JSON.stringify(init.body ?? {})),
                                purpose: '',
                                edgeCasesHandled: [],
                                errorHandling: [],
                                detailedLines: [],
                            });
                            if (exported) exports.push({ name, type: 'function', file: normalizedFilePath });
                        } else {
                            const span = getSpan(decl);
                            const line = lineIndex.getLine(span.start);
                            for (const name of variableNames) {
                                if (!name) continue;
                                const variableNode: ParsedVariable = {
                                    id: allocateId('var', name),
                                    name,
                                    type: 'any',
                                    file: normalizedFilePath,
                                    line,
                                    isExported: exported,
                                };
                                variables.push(variableNode);
                                if (exported) exports.push({ name, type: 'variable', file: normalizedFilePath });
                            }
                        }
                    }
                    break;
                }

                // ── Export Default ─────────────────────────────────────────
                case 'ExportDefaultDeclaration': {
                    const decl = node.declaration;
                    if (!decl) break;

                    if (decl.type === 'FunctionDeclaration' || decl.type === 'FunctionExpression' || decl.type === 'ArrowFunctionExpression') {
                        const name = decl.id?.name ?? 'default';
                        const span = getSpan(node);
                        functions.push({
                            id: allocateId('fn', name),
                            name,
                            file: normalizedFilePath,
                            startLine: lineIndex.getLine(span.start),
                            endLine: lineIndex.getLine(span.end),
                            params: extractParams(decl.params?.items ?? decl.params ?? []),
                            returnType: 'any',
                            isExported: true,
                            isAsync: !!decl.async,
                            calls: extractCalls(decl.body ?? decl, lineIndex),
                            hash: hashContent(JSON.stringify(decl.body ?? {})),
                            purpose: '',
                            edgeCasesHandled: [],
                            errorHandling: [],
                            detailedLines: [],
                        });
                        exports.push({ name, type: 'default', file: normalizedFilePath });
                    } else if (decl.type === 'ClassDeclaration' && decl.id) {
                        exports.push({ name: decl.id.name, type: 'default', file: normalizedFilePath });
                    } else if (decl.type === 'Identifier') {
                        exports.push({ name: decl.name, type: 'default', file: normalizedFilePath });
                    }
                    break;
                }

                // ── Named Exports ──────────────────────────────────────────
                case 'ExportNamedDeclaration': {
                    // Re-export specifiers: export { foo, bar }
                    for (const spec of node.specifiers ?? []) {
                        if (spec.exported?.name) {
                            exports.push({ name: spec.exported.name, type: 'variable', file: normalizedFilePath });
                        }
                    }
                    // Declaration is handled by the declaration's own case with parent context
                    break;
                }

                // ── Module-level call expressions ─────────────────────────
                case 'ExpressionStatement': {
                    if (node.expression?.type === 'CallExpression') {
                        const callExpr = node.expression;
                        const calls = extractCalls(callExpr, lineIndex);
                        moduleCalls.push(...calls);
                        
                        // Route detection
                        const callee = callExpr.callee;
                        if (callee && (callee.type === 'StaticMemberExpression' || callee.type === 'MemberExpression')) {
                            const objName = resolveObjectName(callee.object);
                            const propName = resolvePropertyName(callee.property);
                            if (objName && propName && /^(router|app|express|.*[Rr]outer.*)$/i.test(objName) && /^(get|post|put|delete|patch|all)$/i.test(propName)) {
                                const args = callExpr.arguments || [];
                                const pathArg = args[0];
                                if (pathArg && (pathArg.type === 'StringLiteral' || pathArg.type === 'Literal' || pathArg.type === 'TemplateLiteral')) {
                                    const pathVal = pathArg.value || (pathArg.quasis && pathArg.quasis[0]?.value?.raw) || '';
                                    
                                    const handlerArg = args[args.length - 1];
                                    const handlerStr = handlerArg ? content.slice(getSpan(handlerArg).start, getSpan(handlerArg).end).replace(/\s+/g, ' ').trim() : 'unknown';
                                    
                                    const middlewares = args.slice(1, -1).map((a: any) => 
                                        content.slice(getSpan(a).start, getSpan(a).end).replace(/\s+/g, ' ').trim()
                                    );
                                    
                                    routes.push({
                                        method: propName.toUpperCase() as any,
                                        path: String(pathVal),
                                        handler: handlerStr.length > 80 ? handlerStr.slice(0, 80) + '...' : handlerStr,
                                        middlewares,
                                        file: normalizedFilePath,
                                        line: lineIndex.getLine(getSpan(callExpr).start),
                                    });
                                }
                            }
                        }
                    }
                    break;
                }
            }

            // Recurse into children
            for (const key of Object.keys(node)) {
                if (key === 'span' || key === 'type') continue;
                const child = node[key];
                if (Array.isArray(child)) {
                    for (const c of child) {
                        if (c && typeof c === 'object') visit(c, node);
                    }
                } else if (child && typeof child === 'object') {
                    visit(child, node);
                }
            }
        };

        visit(ast);

        return {
            path: normalizedFilePath,
            language: isTS ? 'typescript' : 'javascript',
            functions,
            classes,
            variables,
            generics,
            imports,
            exports,
            routes,
            calls: moduleCalls,
            hash: hashContent(content),
            parsedAt: Date.now(),
        };
    }

    public resolveImports(files: ParsedFile[], projectRoot: string): ParsedFile[] {
        const resolver = new OxcResolver(projectRoot);
        return resolver.resolveBatch(files);
    }

    public getSupportedExtensions(): string[] {
        return ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'];
    }

    private emptyFile(filePath: string, content: string, isTS: boolean): ParsedFile {
        return {
            path: filePath.replace(/\\/g, '/'),
            language: isTS ? 'typescript' : 'javascript',
            functions: [],
            classes: [],
            variables: [],
            generics: [],
            imports: [],
            exports: [],
            routes: [],
            calls: [],
            hash: hashContent(content),
            parsedAt: Date.now(),
        };
    }
}
