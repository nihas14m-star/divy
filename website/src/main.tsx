import { useState } from 'react'
import type { ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleDot,
  Clipboard,
  Code2,
  Database,
  GitBranch,
  GitFork,
  LockKeyhole,
  Network,
  Play,
  ShieldCheck,
  Sparkles,
  Terminal,
  Timer,
  TriangleAlert,
  Workflow,
  Zap,
} from 'lucide-react'
import './styles.css'

const repoUrl = 'https://github.com/nihas14m-star/divy'

type Tab = 'execution' | 'evidence' | 'security'
type EvidenceMode = 'raw' | 'derived'

const pipeline = [
  { label: 'Task', icon: Sparkles },
  { label: 'Profile', icon: CircleDot },
  { label: 'Feasibility', icon: LockKeyhole },
  { label: 'Evidence', icon: Database },
  { label: 'Routing', icon: GitBranch },
  { label: 'Execution', icon: Workflow },
  { label: 'Tools', icon: Terminal },
  { label: 'Verify', icon: ShieldCheck },
  { label: 'Fallback', icon: TriangleAlert },
  { label: 'Memory', icon: Database },
  { label: 'Evaluate', icon: Check },
]

const commands = [
  'python -m pip install -e ".[dev]"',
  'python -m pytest -q',
  'python -m compileall -q divy tests',
  'python -m divy.benchmark',
  'python -m divy.orchestration_benchmark',
  'python -m divy.performance_benchmark',
  'python -m divy.baseline_benchmark',
  'python -m divy.demo',
]

function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('execution')
  const [evidenceMode, setEvidenceMode] = useState<EvidenceMode>('raw')
  const [copied, setCopied] = useState(false)

  const copyCommands = async () => {
    try {
      if (!navigator.clipboard) return
      await navigator.clipboard.writeText(commands.join('\n'))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="site-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="nav-wrap">
        <nav className="nav container" aria-label="Primary navigation">
          <button className="brand" onClick={() => scrollToId('top')} aria-label="Back to top">
            <span className="brand-mark"><span /></span>
            <span>DIVY</span>
          </button>
          <div className="nav-links">
            <button onClick={() => scrollToId('system')}>System</button>
            <button onClick={() => scrollToId('evidence')}>Evidence</button>
            <button onClick={() => scrollToId('security')}>Security</button>
            <button onClick={() => scrollToId('reproduce')}>Reproduce</button>
          </div>
          <a className="nav-github" href={repoUrl} target="_blank" rel="noreferrer">
            <GitFork size={15} /> GitHub <ArrowUpRight size={14} />
          </a>
        </nav>
      </header>

      <main id="top">
        <section className="hero container">
          <div className="hero-copy">
            <div className="eyebrow"><span className="pulse-dot" /> OPEN RESEARCH SYSTEM · VALIDATED BASELINE</div>
            <h1>Orchestration<br /><em>with evidence.</em></h1>
            <p className="hero-lede">DIVY is a provider-neutral AI orchestration system that makes model selection, tool use, verification, recovery, and performance memory observable.</p>
            <div className="hero-actions">
              <a className="button button-primary" href={repoUrl} target="_blank" rel="noreferrer">View on GitHub <ArrowUpRight size={16} /></a>
              <button className="button button-quiet" onClick={() => scrollToId('system')}>Explore architecture <ChevronRight size={16} /></button>
            </div>
            <div className="hero-meta">
              <span><Check size={14} /> 191 tests passing</span>
              <span><CircleDot size={14} /> provider-neutral core</span>
              <span><LockKeyhole size={14} /> hosted unsafe tools fail closed</span>
            </div>
          </div>
          <HeroTopology />
        </section>

        <section className="ticker-strip" aria-label="Project principles">
          <div className="ticker-track">
            <span>MODEL REGISTRY</span><i /> <span>DETERMINISTIC ROUTING</span><i /> <span>VERIFICATION FIRST</span><i /> <span>RAW MEASUREMENTS</span><i /> <span>NO FABRICATED EVIDENCE</span><i />
          </div>
        </section>

        <section className="section container problem-section">
          <SectionIntro index="01" kicker="THE PROBLEM" title={<>Model selection is a<br /><span>systems question.</span></>} copy="Static capability tables are useful, but incomplete. DIVY treats routing as an empirical loop: select a capable model, execute through controlled tools, verify the artifact, then preserve what actually happened." />
          <div className="problem-grid">
            <div className="problem-statement">
              <div className="quote-mark">“</div>
              <p>Choosing an AI model for every task should not depend purely on static assumptions.</p>
              <span className="small-note">The router observes outcomes without turning them into a fake intelligence score.</span>
            </div>
            <div className="signal-list">
              {['Capability fit', 'Execution outcome', 'Independent verification', 'Recovery + fallback', 'Latency + tokens', 'Confidence + recency'].map((item, i) => (
                <div className="signal-row" key={item}><span>0{i + 1}</span><strong>{item}</strong><ArrowUpRight size={15} /></div>
              ))}
            </div>
          </div>
        </section>

        <section className="section system-section" id="system">
          <div className="container">
            <SectionIntro index="02" kicker="THE SYSTEM" title={<>A control loop built<br /><span>to be inspected.</span></>} copy="Every task becomes a provider-neutral profile before routing: capabilities, context, tools, risk, latency, and verification requirements. Every handoff has a contract. Every final claim is checked against evidence, not the model's confidence." />
            <div className="architecture-panel">
              <div className="panel-topline"><span>DIVY / ORCHESTRATION GRAPH</span><span className="panel-live"><span className="pulse-dot" /> LOCAL / DETERMINISTIC</span></div>
              <div className="pipeline-grid">
                {pipeline.map((node, i) => {
                  const Icon = node.icon
                  return <div className="pipeline-node-wrap" key={node.label}>
                    <div className="pipeline-node"><Icon size={18} /><span>{node.label}</span><small>{String(i + 1).padStart(2, '0')}</small></div>
                    {i < pipeline.length - 1 && <div className="pipeline-arrow"><span /></div>}
                  </div>
                })}
              </div>
               <div className="architecture-foot"><span>task profile</span><span>→</span><span>feasible candidates</span><span>→</span><span>verified outcome</span><span>→</span><span>future routing evidence</span></div>
            </div>
            <div className="tabbed-explainer">
              <div className="tab-controls" role="tablist" aria-label="System views">
                {(['execution', 'evidence', 'security'] as Tab[]).map((tab) => <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>{tab}</button>)}
              </div>
              <div className="tab-content">
                {activeTab === 'execution' && <TabContent icon={<Workflow />} title="Execution stays bounded." body="AgentSession normalizes model responses into FINAL, TOOL_CALL, or ERROR. ToolRegistry validates the request; the sandbox executes only the registered capability; the observation returns to the model." tags={['max iterations', 'max tool calls', 'recovery limit', 'cost limit']} />}
                {activeTab === 'evidence' && <TabContent icon={<Database />} title="Evidence stays separate from ranking." body="PerformanceMemory is append-only JSONL. Raw observations retain task context, verification, latency, token usage, recovery, contribution, and outcome. Derived routing signals are computed later." tags={['append-only', 'Wilson intervals', 'recency', 'sample shrinkage']} />}
                {activeTab === 'security' && <TabContent icon={<LockKeyhole />} title="The unsafe edge is explicit." body="test.run uses host-process exec() for local fixtures. Hosted adapters reject requests that expose this tool before network access. Path restrictions are not process isolation, so the boundary remains fail-closed." tags={['fail closed', 'no fake sandbox', 'credentials excluded', 'future isolation']} />}
              </div>
            </div>
          </div>
        </section>

        <section className="section routing-section container" id="routing">
          <SectionIntro index="03" kicker="ROUTING" title={<>Route on capability.<br /><span>Learn from outcomes.</span></>} copy="The router filters for feasibility first, then combines static metadata with historical evidence, including verified outcomes where available. It can coordinate primary, fallback, verifier, and synthesis roles, but does not claim to be optimal." />
          <div className="routing-layout">
            <div className="routing-visual">
              <div className="visual-label">CAPABLE CANDIDATES / TASK: SOFTWARE_ENGINEERING</div>
              <div className="model-stack">
                <ModelRow model="bolt-e4b" role="executor / fallback" score="0.579" tone="lime" />
                <ModelRow model="lightning-30b" role="planner / coding" score="0.737" tone="amber" active />
                <ModelRow model="super-vl-120b" role="multimodal verifier" score="—" tone="muted" />
              </div>
              <div className="route-line"><span className="route-source">TASK</span><div className="route-path"><span /><span /><span /></div><span className="route-target">SELECTED MODEL</span></div>
            </div>
            <div className="routing-notes">
              <div className="note-block"><span className="note-index">A</span><div><strong>Static first</strong><p>Capabilities, availability, complexity, known cost and latency define who can participate.</p></div></div>
              <div className="note-block"><span className="note-index">B</span><div><strong>Evidence shrinks</strong><p>Small samples are pulled toward a prior. Recent outcomes can be weighted without mutating history.</p></div></div>
              <div className="note-block"><span className="note-index">C</span><div><strong>Fallback is observable</strong><p>When verification fails, a bounded alternate attempt is recorded as part of the result.</p></div></div>
            </div>
          </div>
          <div className="evidence-toggle" role="group" aria-label="Evidence view">
            <div className="toggle-copy"><span>ROUTING INPUTS</span><strong>{evidenceMode === 'raw' ? 'Raw measurements' : 'Derived evidence'}</strong></div>
            <div className="segmented"><button className={evidenceMode === 'raw' ? 'active' : ''} onClick={() => setEvidenceMode('raw')}>Raw</button><button className={evidenceMode === 'derived' ? 'active' : ''} onClick={() => setEvidenceMode('derived')}>Derived</button></div>
            <div className="evidence-sample">{evidenceMode === 'raw' ? <><code>provider_success: true</code><code>verification: true</code><code>latency_ms: null</code><code>sample_size: 4</code></> : <><code>confidence: weak</code><code>recency_weight: 0.30</code><code>shrinkage: 0.286</code><code>rank: evidence-only</code></>}</div>
          </div>
        </section>

        <section className="section memory-section">
          <div className="container split-section">
            <div><SectionIntro index="04" kicker="PERFORMANCE MEMORY" title={<>History is a<br /><span>first-class artifact.</span></>} copy="Append-only local JSONL keeps raw observations traceable, reloadable, and separate from static provider metadata." /></div>
            <MemoryCard />
          </div>
        </section>

        <section className="section evaluation-section container" id="evidence">
          <SectionIntro index="05" kicker="EVALUATION" title={<>No single<br /><span>intelligence score.</span></>} copy="DIVY preserves the measurements that make an execution explainable: what ran, what failed, what verified, and what contributed." />
          <div className="metric-grid">
            <Metric label="EXECUTION ID" value="traceable" detail="per model call / task" />
            <Metric label="QUALITY" value="verified" detail="independent artifact check" />
            <Metric label="EFFICIENCY" value="measured" detail="latency · tokens · tools" />
            <Metric label="ECONOMICS" value="honest" detail="null when unavailable" />
          </div>
          <div className="evaluation-table">
            {['Task success', 'Verification passed', 'False acceptance', 'Model contribution', 'Recovery attempts', 'Historical evidence'].map((item, i) => <div key={item} className="eval-row"><span>0{i + 1}</span><strong>{item}</strong><span className="eval-value">{['boolean', 'boolean | null', 'boolean', 'per call', 'integer', 'pre-run only'][i]}</span><ArrowUpRight size={14} /></div>)}
          </div>
        </section>

        <section className="section results-section" id="results">
          <div className="container">
            <SectionIntro index="06" kicker="EXPERIMENTS / RESULTS" title={<>Evidence before<br /><span>enthusiasm.</span></>} copy="The validated baseline measures orchestration mechanics and controlled routing behavior. It does not prove model intelligence or dynamic-routing superiority." />
            <div className="result-stat-grid">
              <ResultStat value="191/191" label="repository tests" />
              <ResultStat value="20/20" label="B benchmarks" />
              <ResultStat value="10/10" label="O benchmarks" />
              <ResultStat value="10/10" label="P benchmarks" />
              <ResultStat value="9/9" label="scripted RW tasks" />
            </div>
            <div className="research-card">
              <div className="research-heading"><span className="status-tag">CONCLUSION / CURRENT CONFIGURATION</span><h3>Dynamic routing has not demonstrated a measurable advantage over simpler strategies.</h3></div>
              <div className="research-columns">
                <div><span className="column-label">COLD-START</span><p>Fixed routing matched 100% success and was faster than dynamic routing in the tested pilot.</p></div>
                <div><span className="column-label">WARM-START</span><p>Dynamic routing remained operational, but the sample was small and did not establish superiority.</p></div>
                <div><span className="column-label">CORRECTED PILOT</span><p>Fallback worked once for RW06. Historical evidence is preserved, not cherry-picked.</p></div>
              </div>
              <div className="research-footer"><TriangleAlert size={16} /> Small samples · limited live tasks · unknown NVIDIA pricing · nondeterministic hosted models</div>
            </div>
          </div>
        </section>

        <section className="section security-section container" id="security">
          <SectionIntro index="07" kicker="SECURITY BOUNDARY" title={<>Explicit limits are<br /><span>part of the design.</span></>} copy="DIVY does not pretend that a path restriction makes host-process Python execution safe. The unsafe boundary is visible, tested, and fail-closed." />
          <div className="security-layout">
            <div className="security-flow">
              <SecurityStep label="HOSTED MODEL" icon={<Network />} tone="neutral" />
              <div className="security-connector"><span /></div>
              <SecurityStep label="HTTP ADAPTER" icon={<Code2 />} tone="neutral" />
              <div className="security-connector blocked"><span>BLOCK</span></div>
              <SecurityStep label="test.run / exec()" icon={<TriangleAlert />} tone="danger" />
            </div>
            <div className="security-copy"><div className="danger-label"><span /> FAIL-CLOSED</div><h3>Path restrictions are not process isolation.</h3><p>Hosted requests that expose <code>test.run</code> are rejected before network access. Local deterministic fixtures remain available for harness validation. Real hosted repository execution requires a future isolated runner.</p><div className="future-chips"><span>gVisor</span><span>Firecracker</span><span>separate container</span></div></div>
          </div>
        </section>

        <section className="section reproduce-section" id="reproduce">
          <div className="container reproduce-layout">
            <div><SectionIntro index="08" kicker="REPRODUCIBILITY" title={<>Run the loop<br /><span>without credits.</span></>} copy="Start with the deterministic local demo. It exercises routing, tools, verification, performance observation, memory update, and evaluation in one command." /><a className="button button-primary" href={repoUrl} target="_blank" rel="noreferrer">Read the full README <ArrowUpRight size={16} /></a></div>
            <div className="terminal-card"><div className="terminal-bar"><span className="terminal-dots"><i /><i /><i /></span><span>divy / local</span><button onClick={copyCommands} aria-label="Copy reproduction commands">{copied ? <Check size={15} /> : <Clipboard size={15} />}</button></div><pre><span className="prompt">$</span> python -m pip install -e ".[dev]"{ '\n' }<span className="prompt">$</span> python -m pytest -q{ '\n' }<span className="prompt">$</span> python -m divy.demo{ '\n' }<span className="output">✓ route → tools → verify → memory → report</span>{ '\n' }<span className="prompt">$</span> python -m divy.smoke{ '\n' }<span className="output dim">↳ safe no-network skip when unconfigured</span></pre></div>
          </div>
        </section>

        <section className="section roadmap-section container">
          <div className="roadmap-heading"><div><SectionIntro index="09" kicker="FUTURE WORK" title={<>The next frontier is<br /><span>more evidence.</span></>} copy="Not promises. Concrete research and engineering questions that remain after the validated baseline." /></div><div className="future-stamp"><Zap size={18} /><span>HONESTLY<br />INCOMPLETE</span></div></div>
          <div className="roadmap-grid">{['Real process isolation for hosted code execution', 'Larger live samples and broader task coverage', 'More robust behavioral verification', 'Empirical routing calibration without metric tuning', 'Additional provider integrations where appropriate'].map((item, i) => <div className="roadmap-item" key={item}><span>0{i + 1}</span><strong>{item}</strong><ArrowUpRight size={15} /></div>)}</div>
        </section>

        <section className="closing-cta container"><div className="closing-mark"><span /><span /><span /></div><p>DIVY / PROVIDER-NEUTRAL AI ORCHESTRATION + EVALUATION</p><h2>Inspect the system.<br /><em>Run the evidence.</em></h2><a className="button button-primary" href={repoUrl} target="_blank" rel="noreferrer"><GitFork size={17} /> Open repository <ArrowUpRight size={16} /></a></section>
      </main>

      <footer className="footer container"><div className="footer-brand"><span className="brand-mark"><span /></span><strong>DIVY</strong><span>Provider-neutral AI orchestration and evaluation.</span></div><div className="footer-links"><button onClick={() => scrollToId('top')}>Back to top <ArrowUp size={14} /></button><a href={repoUrl} target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={14} /></a></div></footer>
    </div>
  )
}

function SectionIntro({ index, kicker, title, copy }: { index: string; kicker: string; title: ReactNode; copy: string }) {
  return <div className="section-intro"><div className="section-index">{index}</div><div><div className="eyebrow">{kicker}</div><h2>{title}</h2><p>{copy}</p></div></div>
}

function HeroTopology() {
  return <div className="hero-visual"><div className="visual-chrome"><span>DIVY / LIVE TRACE</span><span><span className="pulse-dot" /> OBSERVABLE</span></div><div className="topology-canvas"><svg className="topology-lines" viewBox="0 0 560 430" preserveAspectRatio="none" aria-hidden="true"><path d="M70 82 C170 82 188 145 280 184" /><path d="M70 215 C164 215 192 209 280 209" /><path d="M70 350 C170 350 190 274 280 237" /><path d="M280 209 C354 209 380 150 478 114" /><path d="M280 209 C366 209 395 209 478 209" /><path d="M280 209 C354 209 380 270 478 306" /></svg><div className="topo-input topo-one"><span className="model-glyph">M</span><div><strong>capability</strong><small>registry</small></div></div><div className="topo-input topo-two"><span className="model-glyph">E</span><div><strong>evidence</strong><small>performance memory</small></div></div><div className="topo-input topo-three"><span className="model-glyph">R</span><div><strong>recovery</strong><small>fallback path</small></div></div><div className="topo-core"><div className="core-ring"><span>DIVY</span></div><small>ORCHESTRATOR</small></div><div className="topo-output output-one"><span className="mini-icon"><Terminal size={13} /></span><div><strong>tools</strong><small>registered side effects</small></div></div><div className="topo-output output-two"><span className="mini-icon"><ShieldCheck size={13} /></span><div><strong>verify</strong><small>independent outcome</small></div></div><div className="topo-output output-three"><span className="mini-icon"><Database size={13} /></span><div><strong>report</strong><small>raw + derived evidence</small></div></div></div><div className="visual-footer"><span>task_id: RW01</span><span>mode: scripted</span><span>cost: null</span></div></div>
}

function TabContent({ icon, title, body, tags }: { icon: ReactNode; title: string; body: string; tags: string[] }) {
  return <div className="tab-panel"><div className="tab-icon">{icon}</div><div><h3>{title}</h3><p>{body}</p><div className="tag-row">{tags.map(tag => <span key={tag}>{tag}</span>)}</div></div></div>
}

function ModelRow({ model, role, score, tone, active = false }: { model: string; role: string; score: string; tone: string; active?: boolean }) {
  return <div className={`model-row ${active ? 'selected' : ''}`}><span className={`model-led ${tone}`} /><div className="model-info"><strong>nvidia/{model}</strong><small>{role}</small></div><div className="model-score"><span>{score}</span><div className="score-track"><i style={{ width: score === '—' ? '18%' : `${Number(score) * 100}%` }} /></div></div>{active && <span className="selected-label">candidate</span>}</div>
}

function MemoryCard() {
  return <div className="memory-card"><div className="memory-top"><span>performance_memory.jsonl</span><span>append-only</span></div><pre><span className="json-brace">{'{'}</span>{'\n'}  <span className="json-key">"model_id"</span>: <span className="json-string">"scripted-rw01"</span>,{ '\n' }  <span className="json-key">"task_type"</span>: <span className="json-string">"software_engineering"</span>,{ '\n' }  <span className="json-key">"provider_success"</span>: <span className="json-bool">true</span>,{ '\n' }  <span className="json-key">"verification_passed"</span>: <span className="json-bool">true</span>,{ '\n' }  <span className="json-key">"latency_ms"</span>: <span className="json-number">1.139</span>,{ '\n' }  <span className="json-key">"token_count"</span>: <span className="json-number">18</span>,{ '\n' }  <span className="json-key">"recovery_attempts"</span>: <span className="json-number">0</span>,{ '\n' }  <span className="json-key">"final_outcome"</span>: <span className="json-string">"success"</span>{'\n'}<span className="json-brace">{'}'}</span></pre><div className="memory-bottom"><span><span className="pulse-dot" /> observation written</span><span>raw record / not a rank</span></div></div>
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
}

function ResultStat({ value, label }: { value: string; label: string }) {
  return <div className="result-stat"><strong>{value}</strong><span>{label}</span></div>
}

function SecurityStep({ label, icon, tone }: { label: string; icon: ReactNode; tone: string }) {
  return <div className={`security-step ${tone}`}><span>{icon}</span><strong>{label}</strong></div>
}

export default App

createRoot(document.getElementById('root')!).render(<App />)
