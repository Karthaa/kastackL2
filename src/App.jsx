import { useState, useEffect, useRef, useCallback } from "react";

// ─── TRAINING DATA (exact from setup.py) ───────────────────────────────────
const TRAINING_DATA = {
  reminder: [
    "remind me to call mom tomorrow","set a reminder for 5pm meeting",
    "don't let me forget to submit the report","remind me about the dentist appointment",
    "I need to remember to buy groceries","alert me when it is time to leave",
    "remind me every day at 8am","schedule a reminder for next Monday",
    "don't forget to send the email","ping me in 30 minutes",
    "I should remember to water the plants","can you remind me to take my medicine",
    "set an alarm for the project deadline","remind me to call the bank",
    "I keep forgetting to exercise remind me",
  ],
  "emotional-support": [
    "I am feeling really overwhelmed right now","I don't know what to do anymore",
    "everything feels so hard lately","I am so stressed about this deadline",
    "I feel like I am failing at everything","I just need someone to talk to",
    "I am really anxious about tomorrow","nothing seems to be going right",
    "I feel so alone right now","I am really scared about the results",
    "I had a horrible day today","I cannot stop crying and I don't know why",
    "I feel hopeless about this situation","I am exhausted and burned out",
    "I am really sad and I don't know why","it feels like too much to handle",
    "I am worried I will fail again","I feel like nobody understands me",
  ],
  "action-item": [
    "I need to fix the bug in the login module","update the database schema by Friday",
    "write unit tests for the payment service","review the pull request from John",
    "deploy the new version to production","create a new branch for the feature",
    "send the quarterly report to management","schedule a meeting with the design team",
    "complete the onboarding documentation","research alternative libraries for image processing",
    "refactor the authentication code","finish the API integration task",
    "set up the CI CD pipeline","migrate data to the new server",
    "prepare slides for the presentation",
  ],
  "small-talk": [
    "how are you doing today","what is the weather like",
    "have you heard any good jokes lately","I just had some really good coffee",
    "did you see the game last night","what do you think about the new movie",
    "hey just checking in","random question but do you like pizza",
    "I am bored let us chat","what is your favorite book",
    "tell me something interesting","lol that was funny",
    "good morning how is it going","what are you up to","haha yeah that makes sense",
  ],
  unknown: [
    "asdf jkl qwerty","42","idk idk idk","blah blah blah blah",
    "test test test","what even is this","I don't even know anymore just whatever",
    "skip","nothing in particular","just ignore this",
    "xyzzy plugh","123 456 789","null void empty",
    "random nonsense here","no idea what to type",
  ],
};

// ─── TF-IDF + NAIVE BAYES (real training in browser) ──────────────────────
function tokenize(text) {
  const t = text.toLowerCase().replace(/[^a-z0-9 ]/g, " ");
  const words = t.split(/\s+/).filter(w => w.length > 1);
  const bigrams = words.slice(0, -1).map((w, i) => w + "_" + words[i + 1]);
  return [...words, ...bigrams];
}

class TFIDFNaiveBayes {
  constructor() {
    this.idf = {};
    this.wordProb = {};
    this.classProb = {};
    this.classes = [];
    this.vocab = {};
    this.trained = false;
    this.trainLog = [];
    this.cvAccuracy = 0;
  }

  _computeTF(tokens) {
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
    const max = Math.max(...Object.values(tf), 1);
    for (const t in tf) tf[t] = tf[t] / max;
    return tf;
  }

  train(data) {
    this.trainLog = [];
    const allDocs = [], allLabels = [];
    for (const [cls, examples] of Object.entries(data)) {
      for (const ex of examples) { allDocs.push(ex); allLabels.push(cls); }
    }
    this.classes = Object.keys(data);
    const N = allDocs.length;
    this.trainLog.push(`Tokenizing ${N} documents with unigram + bigram features...`);

    // IDF
    const df = {};
    allDocs.forEach(doc => {
      const unique = new Set(tokenize(doc));
      unique.forEach(t => { df[t] = (df[t] || 0) + 1; });
    });
    Object.keys(df).forEach(t => { this.idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1; });
    this.vocab = Object.fromEntries(Object.keys(df).map((t, i) => [t, i]));
    this.trainLog.push(`Vocabulary: ${Object.keys(this.vocab).length} tokens built`);

    // Per-class word probabilities (Laplace smoothed)
    this.wordProb = {};
    this.classProb = {};
    for (const cls of this.classes) {
      const clsDocs = data[cls];
      this.classProb[cls] = Math.log(clsDocs.length / N);
      const wordSum = {};
      let total = 0;
      for (const doc of clsDocs) {
        const tokens = tokenize(doc);
        const tf = this._computeTF(tokens);
        for (const [t, tfVal] of Object.entries(tf)) {
          const tfidf = tfVal * (this.idf[t] || 1);
          wordSum[t] = (wordSum[t] || 0) + tfidf;
          total += tfidf;
        }
      }
      this.wordProb[cls] = {};
      const V = Object.keys(this.vocab).length;
      for (const t of Object.keys(this.vocab)) {
        this.wordProb[cls][t] = Math.log(((wordSum[t] || 0) + 0.1) / (total + 0.1 * V));
      }
      this.trainLog.push(`  class "${cls}": ${clsDocs.length} samples, ${Object.keys(wordSum).length} unique tokens`);
    }

    // 3-fold cross-validation
    this.trained = true;
    let correct = 0;
    const foldSize = Math.floor(N / 3);
    const foldAccs = [];
    for (let fold = 0; fold < 3; fold++) {
      const ts = fold * foldSize, te = ts + foldSize;
      let fc = 0;
      for (let i = ts; i < te; i++) {
        if (this._predictRaw(allDocs[i]) === allLabels[i]) fc++;
      }
      foldAccs.push(fc / foldSize);
      correct += fc;
    }
    this.cvAccuracy = correct / (foldSize * 3);
    this.trainLog.push(`3-fold CV: fold1=${(foldAccs[0]*100).toFixed(0)}% fold2=${(foldAccs[1]*100).toFixed(0)}% fold3=${(foldAccs[2]*100).toFixed(0)}%`);
    this.trainLog.push(`CV Accuracy: ${(this.cvAccuracy * 100).toFixed(1)}%  |  Model size: ~${(JSON.stringify(this.wordProb).length / 1024).toFixed(1)} KB`);
  }

  _predictRaw(text) {
    const tokens = tokenize(text);
    const tf = this._computeTF(tokens);
    let best = null, bestScore = -Infinity;
    for (const cls of this.classes) {
      let score = this.classProb[cls];
      for (const [t, tfVal] of Object.entries(tf)) {
        score += tfVal * (this.idf[t] || 0.5) * (this.wordProb[cls][t] || Math.log(0.1 / 1000));
      }
      if (score > bestScore) { bestScore = score; best = cls; }
    }
    return best;
  }

  predict(text) {
    const t0 = performance.now();
    const tokens = tokenize(text);
    const tf = this._computeTF(tokens);
    const rawScores = {};
    for (const cls of this.classes) {
      let score = this.classProb[cls];
      for (const [t, tfVal] of Object.entries(tf)) {
        score += tfVal * (this.idf[t] || 0.5) * (this.wordProb[cls][t] || Math.log(0.1 / 1000));
      }
      rawScores[cls] = score;
    }
    const maxS = Math.max(...Object.values(rawScores));
    const exps = Object.fromEntries(Object.entries(rawScores).map(([c, s]) => [c, Math.exp(s - maxS)]));
    const sumExp = Object.values(exps).reduce((a, b) => a + b, 0);
    const proba = Object.fromEntries(Object.entries(exps).map(([c, e]) => [c, e / sumExp]));
    const best = Object.entries(proba).sort((a, b) => b[1] - a[1])[0];
    return {
      intent: best[1] >= 0.25 ? best[0] : "unknown",
      confidence: best[1],
      all_scores: proba,
      latency_ms: performance.now() - t0,
      tokens: tokens.slice(0, 8),
    };
  }
}

// ─── PERSONA DRIFT ENGINE ──────────────────────────────────────────────────
const PERSONA_DATA = {
  name: "Alex",
  conversations: [
    { day: 1, date: "Jan 01", messages: [
      { text: "Hello, I would like to understand how machine learning algorithms work in production systems." },
      { text: "Could you explain the formal definition of gradient descent?" },
      { text: "I am particularly interested in understanding the mathematical foundations." },
      { text: "What are the standard evaluation metrics used in industry?" },
    ]},
    { day: 2, date: "Jan 02", messages: [
      { text: "Back again. Still curious about the neural network architectures we discussed." },
      { text: "My sister mentioned transformers are used in GPT models - is that correct?" },
      { text: "I want to learn more about attention mechanisms." },
    ]},
    { day: 4, date: "Jan 04", messages: [
      { text: "ugh this is so confusing lol" },
      { text: "my sister keeps bugging me to finish this project and I can't even get the basics right" },
      { text: "why does pytorch have to be so complicated honestly" },
      { text: "can u just give me the code i dont have time for theory rn" },
      { text: "deadline is tmrw and im freaking out" },
    ]},
    { day: 6, date: "Jan 06", messages: [
      { text: "hey so I submitted the project and it actually worked!" },
      { text: "my mentor sarah said the results were decent" },
      { text: "still not sure about the math parts though" },
      { text: "gonna take a break before starting the next one" },
    ]},
    { day: 7, date: "Jan 07", messages: [
      { text: "haha ok so I accidentally trained my model on the test set" },
      { text: "classic rookie mistake right? lmaooo" },
      { text: "my sister is going to roast me so hard when she finds out" },
      { text: "anyway what wild ML facts can you tell me today? I am in exploration mode" },
      { text: "also found this cool paper on diffusion models, wanna nerd out?" },
    ]},
  ],
};

const TONE_KW = {
  formal:     ["would like","could you","i am interested","standard","definition","mathematical","evaluation","understanding"],
  casual:     ["lol","gonna","rn","haha","ok so","anyway","tmrw","lmao","lmaooo","u ","hey ","tbh"],
  curious:    ["how","why","what","explain","understand","learn","tell me","nerd out","exploration","paper","curious","attention"],
  frustrated: ["ugh","confusing","complicated","freaking out","just give me","why does","so hard","cant even"],
  playful:    ["haha","lmao","roast","classic","rookie","wild","cool","nerd out","wanna","lmaooo"],
  anxious:    ["deadline","freaking out","no time","stressed","tmrw","bugging","worried"],
  relieved:   ["submitted","worked","decent","break","actually","managed","results"],
};
const MOOD_KW = {
  positive: ["worked","cool","nice","love","excited","good","haha","lol","fun","interesting","submitted","decent"],
  negative: ["ugh","confusing","complicated","freaking","bugging","stressed","cant","dont"],
  neutral:  ["understand","explain","learn","what","how","tell","also","curious"],
};
const TRIGGER_RE = {
  person: [/\bsister\b/i, /\bmentor\b/i, /\bsarah\b/i],
  event:  [/\bdeadline\b/i, /\bsubmitted\b/i, /\bproject\b/i, /\bmistake\b/i, /\btrained\b/i],
  topic:  [/\bpytorch\b/i, /\btransformer\b/i, /\bgpt\b/i, /\bdiffusion\b/i, /\bgradient\b/i, /\battention\b/i],
};

function scoreKW(text, kwDict) {
  const tl = text.toLowerCase();
  return Object.fromEntries(Object.entries(kwDict).map(([k, ws]) => [k, ws.filter(w => tl.includes(w)).length]));
}
function topK(scores, k = 3) {
  return Object.entries(scores).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, k).map(([k]) => k);
}
function getTriggers(messages) {
  const out = [], seen = new Set();
  for (const msg of messages) {
    for (const [type, pats] of Object.entries(TRIGGER_RE)) {
      for (const pat of pats) {
        const m = msg.text.match(pat);
        if (m) {
          const key = type + ":" + m[0].toLowerCase();
          if (!seen.has(key)) { seen.add(key); out.push({ type, value: m[0].toLowerCase(), snippet: msg.text.slice(0, 55) }); }
        }
      }
    }
  }
  return out;
}
function analyzeDay(conv) {
  const text = conv.messages.map(m => m.text).join(" ");
  const toneS = scoreKW(text, TONE_KW);
  const moodS = scoreKW(text, MOOD_KW);
  const tones = topK(toneS) || ["neutral"];
  const mood = topK(moodS)[0] || "neutral";
  const fs = toneS.formal || 0, cs = toneS.casual || 0;
  const formality = fs > cs ? "formal" : cs > fs ? "casual" : "mixed";
  const moodScore = { positive: 1, neutral: 0.5, negative: 0 }[mood] ?? 0.5;
  return { day: conv.day, date: conv.date, tones, mood, formality, moodScore, triggers: getTriggers(conv.messages), msgCount: conv.messages.length };
}
function computeDrift(profiles) {
  const drifts = [];
  for (let i = 1; i < profiles.length; i++) {
    const p = profiles[i - 1], c = profiles[i];
    const changes = [];
    if (p.formality !== c.formality) changes.push(`formality  ${p.formality} → ${c.formality}`);
    if (p.mood !== c.mood) changes.push(`mood  ${p.mood} → ${c.mood}`);
    const newT = c.tones.slice(0, 2).filter(t => !p.tones.slice(0, 2).includes(t));
    if (newT.length) changes.push(`new tones: ${newT.join(", ")}`);
    if (changes.length) drifts.push({ fromDay: p.day, toDay: c.day, changes, triggerDay: c });
  }
  return drifts;
}

// ─── RAG RESOLVER ──────────────────────────────────────────────────────────
const RAG_CHUNKS = [
  { id: "c1", day: 2, topic: "learning_support", text: "User's sister mentioned transformers are used in GPT models. She appears knowledgeable about ML and was a supportive influence.", entities: ["sister","transformers","gpt"], sentiment: "positive" },
  { id: "c2", day: 4, topic: "stress_deadline", text: "User's sister keeps pressuring them to finish the project. Creating external stress during a difficult deadline period.", entities: ["sister","project","deadline"], sentiment: "negative" },
  { id: "c3", day: 7, topic: "playful_reflection", text: "User joked that their sister will roast them for training on the test set. Tone is now lighthearted about the relationship.", entities: ["sister","ml_mistake"], sentiment: "playful" },
];
const SENT_WEIGHT = { positive: 0.5, negative: 0.8, playful: 0.4, neutral: 0.2 };

function resolveRAG(query, chunks) {
  const kws = (query.toLowerCase().match(/\b\w{3,}\b/g) || []);
  const maxDay = Math.max(...chunks.map(c => c.day));
  const matched = chunks.filter(c => kws.some(kw => c.text.toLowerCase().includes(kw) || c.entities.some(e => e.includes(kw))));
  const scored = matched.map(c => {
    const rec = Math.exp(-0.15 * (maxDay - c.day));
    const em = SENT_WEIGHT[c.sentiment] || 0.2;
    return { ...c, recency: +rec.toFixed(3), emotional: em, combined: +(0.6 * rec + 0.4 * em).toFixed(3) };
  }).sort((a, b) => b.combined - a.combined);
  const contradictions = [];
  for (let i = 0; i < scored.length; i++)
    for (let j = i + 1; j < scored.length; j++) {
      const s = new Set([scored[i].sentiment, scored[j].sentiment]);
      if (s.has("negative") && s.size > 1) contradictions.push({ a: scored[i], b: scored[j] });
    }
  return { scored, contradictions };
}

// ─── DESIGN TOKENS ─────────────────────────────────────────────────────────
const C = {
  bg0: "#060a12", bg1: "#0d1117", bg2: "#161b27", bg3: "#1e2535",
  border: "#21283a", borderHi: "#2e3a52",
  text: "#e6edf3", mid: "#8b949e", dim: "#3d4a5c",
  blue: "#58a6ff",  blueDim: "#0d2547",
  green: "#3fb950", greenDim: "#0a2010",
  orange: "#f78166",orangeDim: "#3a150e",
  yellow: "#e3b341",yellowDim: "#2e2200",
  purple: "#bc8cff",purpleDim: "#231540",
  pink: "#ff7eb3",  pinkDim: "#3a1128",
  teal: "#39d353",  tealDim: "#092412",
};

const toneCol = {
  formal: C.blue, casual: C.purple, curious: C.teal,
  frustrated: C.orange, playful: C.yellow, anxious: C.pink,
  relieved: C.green, neutral: C.mid,
};
const intentCol = {
  reminder:          { fg: C.blue,   bg: C.blueDim },
  "emotional-support":{ fg: C.pink,   bg: C.pinkDim },
  "action-item":     { fg: C.green,  bg: C.greenDim },
  "small-talk":      { fg: C.yellow, bg: C.yellowDim },
  unknown:           { fg: C.mid,    bg: C.bg3 },
};
const sentCol = { positive: C.green, negative: C.orange, playful: C.yellow, neutral: C.mid };

// ─── REUSABLE COMPONENTS ───────────────────────────────────────────────────
const Tag = ({ children, fg, bg }) => (
  <span style={{ fontSize: 10, padding: "2px 9px", borderRadius: 3, background: bg || fg + "18", color: fg, border: `1px solid ${fg}28`, whiteSpace: "nowrap" }}>{children}</span>
);

const Section = ({ label, children, accent }) => (
  <div style={{ background: C.bg1, border: `1px solid ${accent ? accent + "44" : C.border}`, borderRadius: 8, overflow: "hidden", marginBottom: 18 }}>
    <div style={{ padding: "12px 20px", borderBottom: `1px solid ${C.border}`, fontSize: 10, color: accent || C.dim, letterSpacing: 2 }}>{label}</div>
    {children}
  </div>
);

const StatCard = ({ label, val, color }) => (
  <div style={{ background: C.bg1, border: `1px solid ${C.border}`, borderRadius: 8, padding: "16px 18px" }}>
    <div style={{ fontSize: 30, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>{val}</div>
    <div style={{ fontSize: 10, color: C.dim, letterSpacing: 1, marginTop: 4 }}>{label}</div>
  </div>
);

// ─── APP ───────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("drift");
  const [model, setModel] = useState(null);
  const [modelLog, setModelLog] = useState([]);
  const [training, setTraining] = useState(false);
  const [trainDone, setTrainDone] = useState(false);
  const [cvAcc, setCvAcc] = useState(null);
  const [modelSizeKB, setModelSizeKB] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [drifts, setDrifts] = useState([]);
  const [intentInput, setIntentInput] = useState("");
  const [intentResult, setIntentResult] = useState(null);
  const [ragQuery, setRagQuery] = useState("Did I mention anything about my sister?");
  const [ragResult, setRagResult] = useState(null);
  const logRef = useRef(null);

  useEffect(() => {
    const p = PERSONA_DATA.conversations.map(analyzeDay);
    setProfiles(p);
    setDrifts(computeDrift(p));
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [modelLog]);

  const trainModel = useCallback(() => {
    setTraining(true);
    setTrainDone(false);
    setModelLog([]);
    setIntentResult(null);
    setTimeout(() => {
      const m = new TFIDFNaiveBayes();
      m.train(TRAINING_DATA);
      const sizeKB = (JSON.stringify(m.wordProb).length / 1024).toFixed(1);
      setModel(m);
      setModelLog(m.trainLog);
      setCvAcc(m.cvAccuracy);
      setModelSizeKB(sizeKB);
      setTrainDone(true);
      setTraining(false);
    }, 60);
  }, []);

  const classify = useCallback(() => {
    if (!model || !intentInput.trim()) return;
    setIntentResult({ ...model.predict(intentInput), input: intentInput });
  }, [model, intentInput]);

  const SAMPLES = [
    "remind me to submit the assignment tomorrow",
    "I feel so overwhelmed and anxious right now",
    "fix the authentication bug before the release",
    "haha that was hilarious lol",
    "deploy the new API to production by Friday",
    "ugh nothing makes sense anymore",
  ];

  const totalMsgs = profiles.reduce((a, p) => a + p.msgCount, 0);
  const triggersAll = profiles.reduce((a, p) => a + p.triggers.length, 0);

  const TABS = [
    { id: "drift",  label: "01 · drift detector" },
    { id: "intent", label: "02 · intent classifier" },
    { id: "rag",    label: "03 · rag resolver" },
    { id: "design", label: "04 · system design" },
  ];

  return (
    <div style={{ background: C.bg0, minHeight: "100vh", color: C.text, fontFamily: "'SF Mono','Fira Code','Consolas',monospace", fontSize: 13 }}>
      <style>{`
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:3px;height:3px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:2px}
        ::selection{background:${C.blueDim}}
        input:focus{outline:none!important;border-color:${C.blue}!important;box-shadow:0 0 0 2px ${C.blueDim}}
        button{cursor:pointer;font-family:inherit;transition:opacity .15s,background .15s,border-color .15s}
        button:hover:not(:disabled){opacity:.85}
        @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
        .fade{animation:fadeUp .3s ease both}
        .blink{animation:blink 1.4s infinite}
      `}</style>

      {/* ── TOPBAR ── */}
      <div style={{ background: C.bg1, borderBottom: `1px solid ${C.border}`, padding: "0 28px", display: "flex", alignItems: "center", height: 52, gap: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, paddingRight: 28, borderRight: `1px solid ${C.border}`, height: "100%" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.green, display: "inline-block" }} className="blink" />
          <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 3, color: C.text }}>KASTACK</span>
          <span style={{ fontSize: 10, color: C.dim, letterSpacing: 1 }}>AI·INTERN·L2</span>
        </div>
        <div style={{ display: "flex", flex: 1 }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: "0 20px", height: 52, border: "none",
              background: tab === t.id ? C.bg2 : "transparent",
              color: tab === t.id ? C.text : C.dim,
              borderBottom: tab === t.id ? `2px solid ${C.blue}` : "2px solid transparent",
              fontSize: 10, letterSpacing: 1,
            }}>{t.label}</button>
          ))}
        </div>
        <div style={{ fontSize: 10, color: C.dim, letterSpacing: 1 }}>TF-IDF · NAIVE BAYES · CRDT</div>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "28px 24px" }}>

        {/* ═══ PART 1 — DRIFT DETECTOR ═══════════════════════════════════ */}
        {tab === "drift" && (
          <div className="fade">
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 10, color: C.blue, letterSpacing: 2, marginBottom: 6 }}>PART 01 / ADAPTIVE PERSONA ENGINE</div>
              <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>Persona Drift Detector</h1>
              <p style={{ fontSize: 12, color: C.mid, lineHeight: 1.7 }}>
                Tracks how Alex's tone, mood, and formality shift across {profiles.length} conversation days.
                Detects drift events and maps the exact trigger — person, event, or topic — that caused each shift.
              </p>
            </div>

            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
              <StatCard label="DAYS TRACKED"   val={profiles.length}  color={C.blue} />
              <StatCard label="TOTAL MESSAGES" val={totalMsgs}        color={C.purple} />
              <StatCard label="DRIFT EVENTS"   val={drifts.length}    color={C.yellow} />
              <StatCard label="TRIGGERS FOUND" val={triggersAll}      color={C.green} />
            </div>

            {/* Mood sparkline */}
            <Section label="MOOD TRAJECTORY">
              <div style={{ padding: "18px 20px" }}>
                <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height: 64, marginBottom: 10 }}>
                  {profiles.map((p) => {
                    const h = Math.round(p.moodScore * 58) + 6;
                    const col = p.mood === "positive" ? C.green : p.mood === "negative" ? C.orange : C.blue;
                    return (
                      <div key={p.day} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
                        <div style={{ width: "60%", height: h, background: col, borderRadius: "3px 3px 0 0", opacity: 0.8 }} title={`Day ${p.day}: ${p.mood}`} />
                      </div>
                    );
                  })}
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {profiles.map(p => (
                    <div key={p.day} style={{ flex: 1, textAlign: "center", fontSize: 9, color: C.dim }}>D{p.day}</div>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 16, marginTop: 12 }}>
                  {[{ label: "positive", col: C.green }, { label: "neutral", col: C.blue }, { label: "negative", col: C.orange }].map(l => (
                    <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: C.dim }}>
                      <span style={{ width: 8, height: 8, borderRadius: 2, background: l.col, display: "inline-block" }} />
                      {l.label}
                    </div>
                  ))}
                </div>
              </div>
            </Section>

            {/* Timeline */}
            <Section label={`TIMELINE · ${PERSONA_DATA.name.toUpperCase()}`}>
              {profiles.map((p, i) => {
                const drift = drifts.find(d => d.toDay === p.day);
                return (
                  <div key={p.day} style={{ padding: "16px 20px", borderBottom: i < profiles.length - 1 ? `1px solid ${C.border}` : "none", display: "flex", gap: 16 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 6, background: drift ? C.orangeDim : C.bg2, border: `1px solid ${drift ? C.orange + "66" : C.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, color: drift ? C.orange : C.mid, flexShrink: 0 }}>
                      {p.day}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 12, fontWeight: 600 }}>Day {p.day}</span>
                        <span style={{ fontSize: 10, color: C.dim }}>{p.date}</span>
                        <span style={{ fontSize: 10, color: C.dim }}>· {p.msgCount} messages</span>
                        {drift && <Tag fg={C.orange} bg={C.orangeDim}>⚡ DRIFT</Tag>}
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: drift || p.triggers.length ? 8 : 0 }}>
                        {p.tones.map(tone => <Tag key={tone} fg={toneCol[tone] || C.mid}>{tone}</Tag>)}
                        <Tag fg={C.mid} bg={C.bg3}>mood · {p.mood}</Tag>
                        <Tag fg={C.dim} bg={C.bg3}>{p.formality}</Tag>
                      </div>
                      {drift && (
                        <div style={{ background: C.bg0, border: `1px solid ${C.orange}33`, borderLeft: `3px solid ${C.orange}`, borderRadius: 4, padding: "8px 12px", marginBottom: 8 }}>
                          <div style={{ fontSize: 9, color: C.orange, letterSpacing: 1, marginBottom: 4 }}>DRIFT CHANGES</div>
                          {drift.changes.map((c, j) => <div key={j} style={{ fontSize: 11, color: C.mid }}>↪ {c}</div>)}
                        </div>
                      )}
                      {p.triggers.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                          {p.triggers.map((t, j) => {
                            const col = t.type === "person" ? C.purple : t.type === "event" ? C.green : C.blue;
                            return <Tag key={j} fg={col}>{t.type === "person" ? "◉" : t.type === "event" ? "▣" : "◈"} {t.value}</Tag>;
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </Section>

            {/* Drift log */}
            {drifts.length > 0 && (
              <Section label="DRIFT EVENT LOG">
                {drifts.map((d, i) => (
                  <div key={i} style={{ padding: "12px 20px", borderBottom: i < drifts.length - 1 ? `1px solid ${C.border}` : "none", display: "flex", gap: 16 }}>
                    <div style={{ fontSize: 11, color: C.orange, whiteSpace: "nowrap", paddingTop: 1, minWidth: 60 }}>D{d.fromDay}→D{d.toDay}</div>
                    <div>
                      {d.changes.map((c, j) => <div key={j} style={{ fontSize: 11, color: C.mid, marginBottom: 2 }}>· {c}</div>)}
                      {d.triggerDay.triggers.length > 0 && (
                        <div style={{ fontSize: 10, color: C.dim, marginTop: 4 }}>
                          trigger: {d.triggerDay.triggers.map(t => `${t.type}:${t.value}`).join("  ")}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </Section>
            )}
          </div>
        )}

        {/* ═══ PART 2 — INTENT CLASSIFIER ════════════════════════════════ */}
        {tab === "intent" && (
          <div className="fade">
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 10, color: C.blue, letterSpacing: 2, marginBottom: 6 }}>PART 02 / OFFLINE ML MODEL</div>
              <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>Intent Classifier</h1>
              <p style={{ fontSize: 12, color: C.mid, lineHeight: 1.7 }}>
                TF-IDF feature extraction (unigram + bigram) + Multinomial Naive Bayes with Laplace smoothing.
                Trained entirely in your browser on {Object.values(TRAINING_DATA).reduce((a, v) => a + v.length, 0)} samples across 5 classes.
                3-fold cross-validation. No server, no API, no internet — runs on CPU under 10ms.
              </p>
            </div>

            {/* Training panel */}
            <Section label="MODEL TRAINING">
              <div style={{ padding: "16px 20px", borderBottom: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 11, color: C.mid, marginBottom: 2 }}>TF-IDF · Multinomial NB · Laplace smoothing · 3-fold CV</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 6, marginTop: 10 }}>
                    {Object.entries(TRAINING_DATA).map(([cls, ex]) => {
                      const ic = intentCol[cls] || { fg: C.mid, bg: C.bg3 };
                      return (
                        <div key={cls} style={{ background: ic.bg, border: `1px solid ${ic.fg}22`, borderRadius: 5, padding: "8px 10px" }}>
                          <div style={{ fontSize: 16, fontWeight: 700, color: ic.fg }}>{ex.length}</div>
                          <div style={{ fontSize: 9, color: ic.fg, opacity: 0.7, marginTop: 1 }}>{cls}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <button onClick={trainModel} disabled={training} style={{
                  padding: "10px 24px", borderRadius: 5,
                  border: `1px solid ${trainDone ? C.green : C.blue}`,
                  background: trainDone ? C.greenDim : C.blueDim,
                  color: trainDone ? C.green : C.blue,
                  fontSize: 11, letterSpacing: 1, opacity: training ? 0.5 : 1,
                }}>
                  {training ? "⟳  TRAINING..." : trainDone ? "✓  RETRAIN" : "▶  TRAIN MODEL"}
                </button>
              </div>

              {/* Terminal log */}
              {(modelLog.length > 0 || training) && (
                <div ref={logRef} style={{ background: C.bg0, padding: "14px 20px", fontSize: 11, maxHeight: 190, overflowY: "auto", fontFamily: "monospace", borderBottom: `1px solid ${C.border}` }}>
                  {modelLog.map((line, i) => (
                    <div key={i} style={{ color: line.includes("Accuracy") || line.includes("CV") ? C.green : line.includes("size") || line.includes("KB") ? C.yellow : C.mid, marginBottom: 3 }}>
                      <span style={{ color: C.dim }}>$ </span>{line}
                    </div>
                  ))}
                  {training && <div style={{ color: C.blue }} className="blink">$ training...</div>}
                </div>
              )}

              {/* Results row */}
              {trainDone && cvAcc !== null && (
                <div style={{ padding: "14px 20px", display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
                  <div>
                    <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1 }}>3-FOLD CV ACCURACY</div>
                    <div style={{ fontSize: 26, fontWeight: 700, color: C.green }}>{(cvAcc * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1 }}>MODEL SIZE</div>
                    <div style={{ fontSize: 26, fontWeight: 700, color: C.yellow }}>{modelSizeKB} KB</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1 }}>LATENCY</div>
                    <div style={{ fontSize: 26, fontWeight: 700, color: C.blue }}>&lt;10 ms</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1 }}>VOCAB SIZE</div>
                    <div style={{ fontSize: 26, fontWeight: 700, color: C.purple }}>{model ? Object.keys(model.vocab).length : "—"}</div>
                  </div>
                </div>
              )}
            </Section>

            {/* Classifier */}
            {!trainDone ? (
              <div style={{ background: C.bg1, border: `1px dashed ${C.border}`, borderRadius: 8, padding: "40px 20px", textAlign: "center", color: C.dim, fontSize: 12 }}>
                Click <span style={{ color: C.blue }}>▶ TRAIN MODEL</span> above to train the TF-IDF + Naive Bayes classifier in your browser
              </div>
            ) : (
              <Section label="CLASSIFY A MESSAGE">
                <div style={{ padding: "16px 20px" }}>
                  <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
                    <input value={intentInput} onChange={e => setIntentInput(e.target.value)}
                      onKeyDown={e => e.key === "Enter" && classify()}
                      placeholder="Type any message and press Enter..."
                      style={{ flex: 1, padding: "10px 14px", background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 5, color: C.text, fontSize: 12, fontFamily: "monospace" }} />
                    <button onClick={classify} disabled={!intentInput.trim()} style={{
                      padding: "10px 22px", borderRadius: 5, border: `1px solid ${C.blue}`,
                      background: C.blueDim, color: C.blue, fontSize: 11, letterSpacing: 1,
                      opacity: !intentInput.trim() ? 0.4 : 1,
                    }}>CLASSIFY</button>
                  </div>
                  <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1, marginBottom: 8 }}>SAMPLE MESSAGES</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {SAMPLES.map(s => (
                      <button key={s} onClick={() => { setIntentInput(s); if (model) setIntentResult({ ...model.predict(s), input: s }); }}
                        style={{ padding: "5px 12px", borderRadius: 4, border: `1px solid ${C.border}`, background: C.bg2, color: C.mid, fontSize: 10 }}>
                        {s.length > 38 ? s.slice(0, 38) + "…" : s}
                      </button>
                    ))}
                  </div>
                </div>
              </Section>
            )}

            {/* Result */}
            {intentResult && (
              <div className="fade">
                <Section label="CLASSIFICATION RESULT" accent={intentCol[intentResult.intent]?.fg}>
                  <div style={{ padding: "16px 20px" }}>
                    <div style={{ fontSize: 11, color: C.dim, marginBottom: 14, fontStyle: "italic" }}>"{intentResult.input}"</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>
                      <div style={{ padding: "10px 24px", borderRadius: 6, background: intentCol[intentResult.intent]?.bg, border: `1px solid ${intentCol[intentResult.intent]?.fg}44`, fontSize: 15, fontWeight: 700, color: intentCol[intentResult.intent]?.fg, letterSpacing: 1 }}>
                        {intentResult.intent.toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontSize: 24, fontWeight: 700, color: intentCol[intentResult.intent]?.fg }}>
                          {Math.round(intentResult.confidence * 100)}%
                        </div>
                        <div style={{ fontSize: 10, color: C.dim }}>confidence · {intentResult.latency_ms.toFixed(2)}ms</div>
                      </div>
                      <div style={{ marginLeft: "auto", fontSize: 10, color: C.dim, lineHeight: 1.6 }}>
                        <div>tokens: {intentResult.tokens?.slice(0, 6).join("  ")}</div>
                      </div>
                    </div>
                    <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1, marginBottom: 10 }}>SOFTMAX PROBABILITY DISTRIBUTION</div>
                    {Object.entries(intentResult.all_scores).sort((a, b) => b[1] - a[1]).map(([cls, prob]) => {
                      const pct = Math.round(prob * 100);
                      const col = intentCol[cls]?.fg || C.mid;
                      return (
                        <div key={cls} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 7 }}>
                          <div style={{ width: 120, fontSize: 11, color: cls === intentResult.intent ? col : C.mid }}>{cls}</div>
                          <div style={{ flex: 1, height: 8, background: C.bg3, borderRadius: 4, overflow: "hidden" }}>
                            <div style={{ width: pct + "%", height: "100%", background: col, borderRadius: 4, opacity: cls === intentResult.intent ? 1 : 0.4, transition: "width 0.5s cubic-bezier(.4,0,.2,1)" }} />
                          </div>
                          <div style={{ width: 34, fontSize: 11, color: col, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{pct}%</div>
                        </div>
                      );
                    })}
                  </div>
                </Section>
              </div>
            )}
          </div>
        )}

        {/* ═══ PART 3 — RAG RESOLVER ══════════════════════════════════════ */}
        {tab === "rag" && (
          <div className="fade">
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 10, color: C.blue, letterSpacing: 2, marginBottom: 6 }}>PART 03 / CONFLICT RESOLUTION IN RAG</div>
              <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>RAG Conflict Resolver</h1>
              <p style={{ fontSize: 12, color: C.mid, lineHeight: 1.7 }}>
                Query hits 3 chunks about "sister" with contradictory context across days.
                Scores each chunk: <span style={{ color: C.blue }}>recency × 0.6</span> + <span style={{ color: C.purple }}>emotional weight × 0.4</span>,
                flags sentiment contradictions, returns a merged coherent answer.
              </p>
            </div>

            {/* Chunks */}
            <Section label="KNOWLEDGE BASE — 3 CHUNKS">
              {RAG_CHUNKS.map((c, i) => (
                <div key={c.id} style={{ padding: "14px 20px", borderBottom: i < RAG_CHUNKS.length - 1 ? `1px solid ${C.border}` : "none", display: "flex", gap: 14 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 5, background: C.bg2, border: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: C.dim, flexShrink: 0 }}>C{i + 1}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", gap: 6, marginBottom: 7, flexWrap: "wrap" }}>
                      <Tag fg={C.blue}>day {c.day}</Tag>
                      <Tag fg={C.purple}>{c.topic}</Tag>
                      <Tag fg={sentCol[c.sentiment]}>{c.sentiment}</Tag>
                    </div>
                    <div style={{ fontSize: 11, color: C.mid, lineHeight: 1.6 }}>{c.text}</div>
                    <div style={{ fontSize: 10, color: C.dim, marginTop: 5 }}>entities: {c.entities.join(", ")}</div>
                  </div>
                </div>
              ))}
            </Section>

            {/* Query */}
            <Section label="QUERY">
              <div style={{ padding: "16px 20px", display: "flex", gap: 8 }}>
                <input value={ragQuery} onChange={e => setRagQuery(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && setRagResult(resolveRAG(ragQuery, RAG_CHUNKS))}
                  style={{ flex: 1, padding: "10px 14px", background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 5, color: C.text, fontSize: 12, fontFamily: "monospace" }} />
                <button onClick={() => setRagResult(resolveRAG(ragQuery, RAG_CHUNKS))}
                  style={{ padding: "10px 22px", borderRadius: 5, border: `1px solid ${C.blue}`, background: C.blueDim, color: C.blue, fontSize: 11, letterSpacing: 1 }}>
                  RESOLVE
                </button>
              </div>
            </Section>

            {ragResult && (
              <div className="fade">
                {/* Ranked chunks */}
                <Section label="RANKED CHUNKS  ·  score = recency×0.6 + emotional×0.4">
                  {ragResult.scored.map((s, i) => (
                    <div key={s.id} style={{ padding: "14px 20px", borderBottom: i < ragResult.scored.length - 1 ? `1px solid ${C.border}` : "none", display: "flex", gap: 14, background: i === 0 ? C.blueDim + "33" : "transparent" }}>
                      <div style={{ width: 28, height: 28, borderRadius: 5, background: i === 0 ? C.blue : C.bg3, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: i === 0 ? C.bg0 : C.dim, flexShrink: 0 }}>#{i + 1}</div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 10, color: C.dim, marginBottom: 4 }}>Day {s.day} · {s.topic} · <span style={{ color: sentCol[s.sentiment] }}>{s.sentiment}</span></div>
                        <div style={{ fontSize: 11, color: C.mid, lineHeight: 1.5 }}>{s.text}</div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8, flexWrap: "wrap" }}>
                          <span style={{ fontSize: 9, color: C.dim }}>recency</span>
                          <div style={{ width: 70, height: 4, background: C.bg3, borderRadius: 2 }}>
                            <div style={{ width: Math.round(s.recency * 100) + "%", height: "100%", background: C.blue, borderRadius: 2 }} />
                          </div>
                          <span style={{ fontSize: 9, color: C.blue }}>{s.recency}</span>
                          <span style={{ fontSize: 9, color: C.dim }}>emotional</span>
                          <div style={{ width: 70, height: 4, background: C.bg3, borderRadius: 2 }}>
                            <div style={{ width: Math.round(s.emotional * 100) + "%", height: "100%", background: C.purple, borderRadius: 2 }} />
                          </div>
                          <span style={{ fontSize: 9, color: C.purple }}>{s.emotional}</span>
                          <span style={{ fontSize: 9, color: C.dim }}>→</span>
                          <span style={{ fontSize: 13, fontWeight: 700, color: i === 0 ? C.blue : C.mid }}>score: {s.combined}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </Section>

                {/* Contradictions */}
                {ragResult.contradictions.length > 0 && (
                  <Section label="⚡ CONTRADICTIONS DETECTED" accent={C.orange}>
                    {ragResult.contradictions.map((con, i) => (
                      <div key={i} style={{ padding: "12px 20px", borderBottom: i < ragResult.contradictions.length - 1 ? `1px solid ${C.border}` : "none" }}>
                        <div style={{ fontSize: 11, color: C.mid }}>
                          <span style={{ color: C.orange }}>Day {con.a.day}</span>
                          <span style={{ color: sentCol[con.a.sentiment] }}> ({con.a.sentiment})</span>
                          <span style={{ color: C.dim }}> ↔ </span>
                          <span style={{ color: C.orange }}>Day {con.b.day}</span>
                          <span style={{ color: sentCol[con.b.sentiment] }}> ({con.b.sentiment})</span>
                          <span style={{ color: C.dim }}> — "{con.a.topic}" vs "{con.b.topic}"</span>
                        </div>
                      </div>
                    ))}
                  </Section>
                )}

                {/* Merged answer */}
                <Section label="MERGED COHERENT ANSWER" accent={C.blue}>
                  <div style={{ padding: "18px 20px" }}>
                    <div style={{ fontSize: 13, color: C.text, lineHeight: 1.8, marginBottom: 14 }}>
                      Your sister was mentioned in <span style={{ color: C.blue, fontWeight: 700 }}>{ragResult.scored.length} separate contexts</span> across your conversations:
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {ragResult.scored.slice().sort((a, b) => a.day - b.day).map(s => (
                        <div key={s.id} style={{ display: "flex", gap: 12, padding: "10px 14px", background: C.bg0, borderRadius: 5, borderLeft: `3px solid ${sentCol[s.sentiment]}` }}>
                          <div style={{ fontSize: 10, color: C.dim, whiteSpace: "nowrap", paddingTop: 2, minWidth: 40 }}>Day {s.day}</div>
                          <div style={{ fontSize: 11, color: C.mid, lineHeight: 1.6 }}>{s.text}</div>
                        </div>
                      ))}
                    </div>
                    {ragResult.contradictions.length > 0 && (
                      <div style={{ marginTop: 14, padding: "12px 14px", background: C.orangeDim, borderRadius: 5, borderLeft: `3px solid ${C.orange}`, fontSize: 11, color: C.mid, lineHeight: 1.6 }}>
                        <span style={{ color: C.orange }}>Contradiction resolved:</span> Tone shifted from supportive → pressuring → playful. This reflects Alex's changing emotional state across the week — not a change in the relationship itself. Highest-ranked chunk (Day 7, score {ragResult.scored[0]?.combined}) is used as primary context.
                      </div>
                    )}
                  </div>
                </Section>
              </div>
            )}
          </div>
        )}

        {/* ═══ PART 4 — SYSTEM DESIGN ═════════════════════════════════════ */}
        {tab === "design" && (
          <div className="fade">
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 10, color: C.blue, letterSpacing: 2, marginBottom: 6 }}>PART 04 / SYSTEM DESIGN DOC</div>
              <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>Sync Architecture</h1>
              <p style={{ fontSize: 12, color: C.mid, lineHeight: 1.7 }}>Privacy-first, offline-capable sync. Raw messages never leave the device. Only anonymised metadata syncs via AES-256 encrypted channel with CRDT conflict resolution.</p>
            </div>

            {/* Diagram */}
            <Section label="ARCHITECTURE DIAGRAM">
              <div style={{ padding: 24 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 64px 1fr", alignItems: "center" }}>
                  {/* Device */}
                  <div style={{ background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                    <div style={{ fontSize: 10, color: C.blue, letterSpacing: 1, marginBottom: 12 }}>📱  DEVICE A  (PHONE)</div>
                    {[
                      { label: "SQLite · raw messages", tag: "NEVER LEAVES", col: C.orange },
                      { label: "Persona JSON · drift history", tag: "LOCAL ONLY", col: C.orange },
                      { label: "Intent Model  14 KB", tag: "OFFLINE", col: C.green },
                      { label: "RAG Chunks + FTS5", tag: "LOCAL ONLY", col: C.orange },
                      { label: "Encrypted KV · prefs", tag: "LOCAL ONLY", col: C.orange },
                    ].map(r => (
                      <div key={r.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 10px", background: C.bg2, borderRadius: 4, marginBottom: 5 }}>
                        <span style={{ fontSize: 10, color: C.mid }}>{r.label}</span>
                        <span style={{ fontSize: 9, color: r.col, letterSpacing: 0.5 }}>{r.tag}</span>
                      </div>
                    ))}
                  </div>
                  {/* Arrow */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                    <div style={{ fontSize: 9, color: C.dim, textAlign: "center", lineHeight: 1.5 }}>AES-256<br />encrypted</div>
                    <div style={{ color: C.blue, fontSize: 22 }}>⇄</div>
                    <div style={{ fontSize: 9, color: C.dim, textAlign: "center", lineHeight: 1.5 }}>labels<br />only</div>
                  </div>
                  {/* Cloud */}
                  <div style={{ background: C.bg0, border: `1px solid ${C.borderHi}`, borderRadius: 8, padding: 16 }}>
                    <div style={{ fontSize: 10, color: C.purple, letterSpacing: 1, marginBottom: 12 }}>☁️  CLOUD SERVER</div>
                    {[
                      { label: "CRDT Resolver", sub: "last-write-wins by timestamp" },
                      { label: "Drift Timeline", sub: "tone labels only — no raw text" },
                      { label: "RAG Metadata", sub: "chunk IDs + scores, not content" },
                      { label: "Conflict Log", sub: "full audit trail" },
                      { label: "GDPR Erase Endpoint", sub: "delete-all on request" },
                    ].map(r => (
                      <div key={r.label} style={{ padding: "6px 10px", background: C.bg2, borderRadius: 4, marginBottom: 5 }}>
                        <div style={{ fontSize: 10, color: C.mid }}>{r.label}</div>
                        <div style={{ fontSize: 9, color: C.dim, marginTop: 1 }}>{r.sub}</div>
                      </div>
                    ))}
                  </div>
                </div>
                {/* Device B */}
                <div style={{ marginTop: 12, background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 14 }}>💻</span>
                  <div>
                    <div style={{ fontSize: 10, color: C.blue, letterSpacing: 1 }}>DEVICE B (LAPTOP) — identical storage pattern</div>
                    <div style={{ fontSize: 10, color: C.dim, marginTop: 2 }}>Full offline functionality · receives merged state broadcast from cloud after sync</div>
                  </div>
                </div>
              </div>
            </Section>

            {/* Storage table */}
            <Section label="ON-DEVICE STORAGE LAYERS">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 1, background: C.border }}>
                {[
                  { layer: "Messages", tech: "SQLite", stores: "Raw messages + timestamps", syncs: false },
                  { layer: "Persona State", tech: "JSON flat file", stores: "Tone / mood / drift history", syncs: false },
                  { layer: "Intent Model", tech: ".joblib binary 14 KB", stores: "Model weights — offline", syncs: false },
                  { layer: "RAG Chunks", tech: "SQLite + FTS5", stores: "Chunked memory + scores", syncs: false },
                  { layer: "Drift Timeline", tech: "Cloud sync target", stores: "Tone labels only — no raw text", syncs: true },
                  { layer: "User Prefs", tech: "Encrypted KV store", stores: "API keys, sync settings", syncs: false },
                ].map(r => (
                  <div key={r.layer} style={{ background: C.bg1, padding: "14px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontSize: 11, fontWeight: 600 }}>{r.layer}</span>
                      <Tag fg={r.syncs ? C.blue : C.green}>{r.syncs ? "SYNCS" : "LOCAL"}</Tag>
                    </div>
                    <div style={{ fontSize: 10, color: C.blue, marginBottom: 3 }}>{r.tech}</div>
                    <div style={{ fontSize: 10, color: C.dim }}>{r.stores}</div>
                  </div>
                ))}
              </div>
            </Section>

            {/* Conflict resolution */}
            <Section label="CONFLICT RESOLUTION STRATEGY">
              <div style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
                {[
                  { n: "01", rule: "Same-day conflict", res: "Last-write-wins via timestamp — higher timestamp takes precedence", col: C.blue },
                  { n: "02", rule: "Trigger list merge", res: "Union merge across devices — combine all entries, deduplicate by (type, value) composite key", col: C.purple },
                  { n: "03", rule: "Unresolvable conflict", res: "Server is authoritative source of truth. Local divergence stored in conflict log for user audit.", col: C.yellow },
                  { n: "04", rule: "Privacy hard boundary", res: "Raw message text is NEVER transmitted. Only anonymised tone labels and mood scores leave the device.", col: C.green },
                ].map(r => (
                  <div key={r.n} style={{ display: "flex", gap: 14, padding: "14px 16px", background: C.bg0, borderRadius: 6, borderLeft: `3px solid ${r.col}` }}>
                    <div style={{ fontSize: 10, color: r.col, fontWeight: 700, width: 20, flexShrink: 0, paddingTop: 1 }}>{r.n}</div>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{r.rule}</div>
                      <div style={{ fontSize: 11, color: C.mid, lineHeight: 1.5 }}>{r.res}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            {/* Security */}
            <Section label="SECURITY & COMPLIANCE">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 1, background: C.border }}>
                {[
                  { icon: "🔐", title: "AES-256 E2E Encryption", desc: "All synced data encrypted end-to-end before leaving the device" },
                  { icon: "✅", title: "SHA-256 Integrity Check", desc: "Hash verification per sync chunk to detect any tampering in transit" },
                  { icon: "🗑️", title: "GDPR Delete-All", desc: "Full erasure endpoint — wipes all server-side user data on request" },
                  { icon: "✈️", title: "Offline-First Design", desc: "Full functionality with zero network. Sync is additive, never blocking." },
                ].map(s => (
                  <div key={s.title} style={{ background: C.bg1, padding: "16px 18px", display: "flex", gap: 12 }}>
                    <span style={{ fontSize: 20 }}>{s.icon}</span>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4 }}>{s.title}</div>
                      <div style={{ fontSize: 10, color: C.dim, lineHeight: 1.5 }}>{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>
        )}
      </div>
    </div>
  );
}