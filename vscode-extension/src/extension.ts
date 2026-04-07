import * as vscode from 'vscode';

const VULNERABILITY_PATTERNS: Record<string, { pattern: RegExp; severity: string; description: string }[]> = {
  'SQL Injection': [
    { pattern: /(\bexec\b|\bexecute\b).*\+.*\b(req|request|params|query|input|user)\b/gi, severity: 'CRITICAL', description: 'String concatenation in SQL query — use parameterized queries instead' },
    { pattern: /f["'].*SELECT.*\{.*\}/gi, severity: 'CRITICAL', description: 'F-string in SQL query — vulnerable to SQL injection' },
    { pattern: /\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE|DROP)/gi, severity: 'CRITICAL', description: '.format() in SQL query — use parameterized queries' },
    { pattern: /query\s*\(\s*[`"'].*\$\{/gi, severity: 'CRITICAL', description: 'Template literal in database query — use prepared statements' },
  ],
  'Cross-Site Scripting (XSS)': [
    { pattern: /innerHTML\s*=\s*(?!['"]<)/gi, severity: 'HIGH', description: 'Direct innerHTML assignment — use textContent or sanitize input' },
    { pattern: /document\.write\s*\(/gi, severity: 'HIGH', description: 'document.write() is dangerous — can enable XSS attacks' },
    { pattern: /dangerouslySetInnerHTML/gi, severity: 'MEDIUM', description: 'dangerouslySetInnerHTML used — ensure input is sanitized' },
    { pattern: /\.html\(\s*(?:req|request|params|query|input|user)/gi, severity: 'HIGH', description: 'User input passed to .html() — sanitize before rendering' },
  ],
  'Command Injection': [
    { pattern: /(?:exec|spawn|system|popen)\s*\(.*\+.*(?:req|request|params|input|user)/gi, severity: 'CRITICAL', description: 'User input in shell command — use argument arrays, not string concatenation' },
    { pattern: /child_process.*exec\s*\(\s*`/gi, severity: 'CRITICAL', description: 'Template literal in child_process.exec — use execFile with argument array' },
    { pattern: /os\.system\s*\(.*(?:input|argv|request)/gi, severity: 'CRITICAL', description: 'User input in os.system() — use subprocess with shell=False' },
    { pattern: /subprocess\.(?:call|run|Popen)\s*\(.*shell\s*=\s*True/gi, severity: 'HIGH', description: 'subprocess with shell=True — avoid when handling user input' },
  ],
  'Hardcoded Secrets': [
    { pattern: /(?:password|passwd|secret|api_key|apikey|token|auth)\s*=\s*["'][^"']{8,}/gi, severity: 'HIGH', description: 'Potential hardcoded secret — use environment variables' },
    { pattern: /(?:AWS_SECRET|PRIVATE_KEY|-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY)/gi, severity: 'CRITICAL', description: 'Private key or AWS secret in source code — remove immediately' },
  ],
  'Insecure Configuration': [
    { pattern: /(?:verify|ssl|tls)\s*=\s*False/gi, severity: 'HIGH', description: 'SSL/TLS verification disabled — vulnerable to MITM attacks' },
    { pattern: /CORS\s*\(\s*\*\s*\)|Access-Control-Allow-Origin.*\*/gi, severity: 'MEDIUM', description: 'Wildcard CORS — restrict to specific origins' },
    { pattern: /DEBUG\s*=\s*True/gi, severity: 'MEDIUM', description: 'Debug mode enabled — disable in production' },
  ],
  'Path Traversal': [
    { pattern: /(?:readFile|readFileSync|open|fopen)\s*\(.*(?:req|request|params|query|input|user)/gi, severity: 'HIGH', description: 'User input in file path — validate and sanitize to prevent path traversal' },
    { pattern: /\.\.\/|\.\.\\|\.\.\%2[fF]/gi, severity: 'MEDIUM', description: 'Potential path traversal pattern detected' },
  ],
};

interface Finding {
  line: number;
  column: number;
  category: string;
  severity: string;
  description: string;
  code: string;
}

function scanCode(text: string): Finding[] {
  const findings: Finding[] = [];
  const lines = text.split('\n');

  for (const [category, patterns] of Object.entries(VULNERABILITY_PATTERNS)) {
    for (const { pattern, severity, description } of patterns) {
      // Reset regex state
      const regex = new RegExp(pattern.source, pattern.flags);

      for (let lineNum = 0; lineNum < lines.length; lineNum++) {
        const line = lines[lineNum];

        // Skip comments
        if (line.trimStart().startsWith('//') || line.trimStart().startsWith('#') || line.trimStart().startsWith('*')) {
          continue;
        }

        let match;
        while ((match = regex.exec(line)) !== null) {
          findings.push({
            line: lineNum + 1,
            column: match.index + 1,
            category,
            severity,
            description,
            code: line.trim(),
          });
        }
      }
    }
  }

  return findings;
}

function formatFindings(findings: Finding[], filename: string): string {
  if (findings.length === 0) {
    return `## GhostLM Security Review\n\n**File:** ${filename}\n\n**No security issues detected.**\n\nThis doesn't guarantee the code is secure — GhostLM checks for common vulnerability patterns. Manual review is always recommended.`;
  }

  const critical = findings.filter(f => f.severity === 'CRITICAL').length;
  const high = findings.filter(f => f.severity === 'HIGH').length;
  const medium = findings.filter(f => f.severity === 'MEDIUM').length;

  let report = `## GhostLM Security Review\n\n`;
  report += `**File:** ${filename}\n`;
  report += `**Issues Found:** ${findings.length}\n`;
  report += `**Breakdown:** ${critical} Critical | ${high} High | ${medium} Medium\n\n`;
  report += `---\n\n`;

  for (const finding of findings) {
    const icon = finding.severity === 'CRITICAL' ? '🔴' : finding.severity === 'HIGH' ? '🟠' : '🟡';
    report += `### ${icon} ${finding.category} [${finding.severity}]\n`;
    report += `**Line ${finding.line}:** ${finding.description}\n`;
    report += `\`\`\`\n${finding.code}\n\`\`\`\n\n`;
  }

  report += `---\n*Scanned by GhostLM Security Review v0.1.0*`;
  return report;
}

async function queryGhostLMAPI(prompt: string): Promise<string | null> {
  const config = vscode.workspace.getConfiguration('ghostlm');
  const endpoint = config.get<string>('apiEndpoint', 'http://localhost:8000');
  const maxTokens = config.get<number>('maxTokens', 200);
  const temperature = config.get<number>('temperature', 0.7);

  try {
    const response = await fetch(`${endpoint}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: maxTokens,
        temperature,
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) return null;
    const data = await response.json() as any;
    return data.text || data.generated_text || null;
  } catch {
    return null;
  }
}

export function activate(context: vscode.ExtensionContext) {
  // Diagnostic collection for inline warnings
  const diagnostics = vscode.languages.createDiagnosticCollection('ghostlm');
  context.subscriptions.push(diagnostics);

  // Command: Review entire file
  const reviewFile = vscode.commands.registerCommand('ghostlm.reviewFile', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('No active file to review.');
      return;
    }

    const text = editor.document.getText();
    const filename = editor.document.fileName.split('/').pop() || 'unknown';

    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'GhostLM: Scanning for security issues...',
      cancellable: false,
    }, async () => {
      const findings = scanCode(text);

      // Set diagnostics for inline markers
      const diags: vscode.Diagnostic[] = findings.map(f => {
        const range = new vscode.Range(f.line - 1, 0, f.line - 1, f.code.length);
        const severity = f.severity === 'CRITICAL'
          ? vscode.DiagnosticSeverity.Error
          : f.severity === 'HIGH'
          ? vscode.DiagnosticSeverity.Warning
          : vscode.DiagnosticSeverity.Information;

        const diag = new vscode.Diagnostic(range, `[${f.category}] ${f.description}`, severity);
        diag.source = 'GhostLM';
        return diag;
      });
      diagnostics.set(editor.document.uri, diags);

      // Show report in new document
      const report = formatFindings(findings, filename);
      const doc = await vscode.workspace.openTextDocument({
        content: report,
        language: 'markdown',
      });
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    });
  });

  // Command: Review selection
  const reviewSelection = vscode.commands.registerCommand('ghostlm.reviewSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) {
      vscode.window.showWarningMessage('Select code to review.');
      return;
    }

    const text = editor.document.getText(editor.selection);
    const filename = editor.document.fileName.split('/').pop() || 'selection';

    const findings = scanCode(text);
    const report = formatFindings(findings, filename);

    const doc = await vscode.workspace.openTextDocument({
      content: report,
      language: 'markdown',
    });
    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
  });

  // Command: Explain vulnerability (uses GhostLM API if running)
  const explainVuln = vscode.commands.registerCommand('ghostlm.explainVulnerability', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) {
      vscode.window.showWarningMessage('Select a code snippet to explain.');
      return;
    }

    const selectedCode = editor.document.getText(editor.selection);
    const prompt = `Explain the security vulnerability in this code and how to fix it:\n\n${selectedCode}\n\nVulnerability analysis:`;

    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'GhostLM: Analyzing vulnerability...',
      cancellable: false,
    }, async () => {
      const aiResponse = await queryGhostLMAPI(prompt);

      let report: string;
      if (aiResponse) {
        report = `## GhostLM Vulnerability Analysis\n\n**Selected Code:**\n\`\`\`\n${selectedCode}\n\`\`\`\n\n**Analysis:**\n${aiResponse}\n\n---\n*Analyzed by GhostLM API*`;
      } else {
        // Fallback: use pattern matching
        const findings = scanCode(selectedCode);
        if (findings.length > 0) {
          report = `## GhostLM Vulnerability Analysis\n\n**Selected Code:**\n\`\`\`\n${selectedCode}\n\`\`\`\n\n`;
          for (const f of findings) {
            report += `**${f.category} [${f.severity}]:** ${f.description}\n\n`;
          }
          report += `\n> GhostLM API not running. Start it with: \`python scripts/api.py\`\n> For AI-powered analysis, ensure the API is accessible.`;
        } else {
          report = `## GhostLM Vulnerability Analysis\n\n**Selected Code:**\n\`\`\`\n${selectedCode}\n\`\`\`\n\n**No known vulnerability patterns detected in selection.**\n\n> For deeper AI analysis, start the GhostLM API: \`python scripts/api.py\``;
        }
      }

      const doc = await vscode.workspace.openTextDocument({
        content: report,
        language: 'markdown',
      });
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    });
  });

  context.subscriptions.push(reviewFile, reviewSelection, explainVuln);

  // Show activation message
  vscode.window.showInformationMessage('GhostLM Security Review is active.');
}

export function deactivate() {}
