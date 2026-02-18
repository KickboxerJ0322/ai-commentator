import express from "express";
import crypto from "crypto";

const app = express();
app.use(express.json({ limit: "10mb" }));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY is required");

const MODEL = process.env.MODEL || "gemini-2.0-flash";

// VOICEVOX（任意）: ローカルで engine を起動している場合に使える
// 例: http://127.0.0.1:50021/docs  :contentReference[oaicite:1]{index=1}
const VOICEVOX_BASE = process.env.VOICEVOX_BASE || "http://127.0.0.1:50021";

// sessionごとの直近コメント（重複回避）
const sessionState = new Map(); // sessionId -> { lastTexts: string[], updatedAt: number }
const SESSION_TTL_MS = 20 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of sessionState.entries()) {
    if (!v?.updatedAt || now - v.updatedAt > SESSION_TTL_MS) sessionState.delete(k);
  }
}, 60 * 1000).unref();

function safeTrimBase64DataUrl(s) {
  const str = String(s || "");
  if (!str) return "";
  return str.includes("base64,") ? str.split("base64,")[1] : str;
}
function estimateBytesFromBase64(b64) {
  if (!b64) return 0;
  const len = b64.length;
  const padding = b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0;
  return Math.floor((len * 3) / 4) - padding;
}
function normalizeForDedup(x) {
  return String(x || "")
    .replace(/[！!。．\s]/g, "")
    .replace(/です|ます|ですね|でしょう|ようです/g, "");
}

function buildGeneralPrompt({ meta, lastTexts }) {
  const prev = (lastTexts || []).slice(-6).join(" / ");

  // クライアント指定の最大文字数（1〜100、デフォルト40）
  const maxChars = Math.max(10, Math.min(100, Number(meta?.maxChars ?? 40)));
  const enableAdv = !!meta?.enableAdvantage;
  const redFeatures = String(meta?.redFeatures || "").trim();
  const blueFeatures = String(meta?.blueFeatures || "").trim();

  return `
あなたはテレビ中継の実況・解説者です。
入力は「直前フレーム」と「現在フレーム」の2枚と補助メタ情報です。
2枚を比較して“変化”を優先してコメントしてください。
変化がない場合、「変化はない」「特に動きはありません」などの文章は出力せず、必ず空文字を返す。

【最重要】出力は必ず JSON のみ。前後に文章やコードブロックは禁止。

出力形式:
{
  "commentary": "1文のみ。最大${maxChars}文字以内。落ち着いた口調。画面に文字が表示されたら、必ず読み上げる。変化がない場合は空文字。",
  "topic": "内容カテゴリ（例: スポーツ/ニュース/ゲーム/会議/自然/その他）",
  "confidence": 0.0-1.0${enableAdv ? `,
  "advantage": {
    "red": 0.0-1.0,
    "blue": 0.0-1.0,
    "reason": "短い理由"
  }` : ""}
}

ルール：
- 1文のみ（改行禁止）
- 煽り・感嘆符の連発禁止
- 見えていない固有名詞は言わない
- 不確実なら confidence を下げる
- 同じ言い回しを連発しない
- 画面に有意な変化が見られない場合は commentary を空文字 "" にする

${enableAdv ? `
【優勢バー（RED/BLUE固定）】
- 2人を固定ラベルで扱う：RED / BLUE
- RED特徴: ${redFeatures || "（未指定）"}
- BLUE特徴: ${blueFeatures || "（未指定）"}
- REDとBLUEを絶対に入れ替えない（左右で固定しない）
- 判別不能なら advantage を出さない（commentaryだけでOK）
- advantage を出す場合、red+blue は概ね 1.0
` : ""}

直近のコメント（重複回避の参考）：
${prev ? prev : "（なし）"}

補助メタ情報：
${JSON.stringify(meta ?? {}, null, 2)}

出力はJSONのみ。
`.trim();
}

async function geminiGenerateTwoImages({ prompt, prevB64, curB64 }) {
  const body = {
    contents: [
      {
        role: "user",
        parts: [
          { text: prompt },
          { inlineData: { mimeType: "image/jpeg", data: prevB64 } },
          { inlineData: { mimeType: "image/jpeg", data: curB64 } },
        ],
      },
    ],
    generationConfig: { temperature: 0.7, maxOutputTokens: 240 },
  };

  const url = `https://generativelanguage.googleapis.com/v1/models/${encodeURIComponent(
    MODEL
  )}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const r = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  const txt = await r.text();
  if (!r.ok) throw new Error(`Gemini API error: ${r.status} ${txt}`);

  const json = JSON.parse(txt);
  const out =
    json?.candidates?.[0]?.content?.parts?.map((p) => p?.text).filter(Boolean).join("") ?? "";
  return out.trim();
}

function tryParseJsonLoose(s) {
  const str = String(s || "");
  const a = str.indexOf("{");
  const b = str.lastIndexOf("}");
  if (a === -1 || b === -1 || b <= a) return null;
  try {
    return JSON.parse(str.slice(a, b + 1));
  } catch {
    return null;
  }
}

function clamp(n, min, max) {
  const x = Number(n);
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, x));
}

// ---- 汎用コメントAPI
app.post("/api/commentary", async (req, res) => {
  try {
    const { imageBase64, prevImageBase64, meta, sessionId } = req.body ?? {};
    const sid = String(sessionId || "").trim() || crypto.randomUUID();

    const curB64 = safeTrimBase64DataUrl(imageBase64);
    const prevB64 = safeTrimBase64DataUrl(prevImageBase64);
    if (!curB64 || !prevB64) {
      return res.status(400).json({ error: "imageBase64 and prevImageBase64 are required", sessionId: sid });
    }

    const maxBytesPerImage = 500 * 1024;
    const curBytes = estimateBytesFromBase64(curB64);
    const prevBytes = estimateBytesFromBase64(prevB64);
    if (curBytes > maxBytesPerImage || prevBytes > maxBytesPerImage) {
      return res.status(413).json({ error: "image too large", detail: { curBytes, prevBytes, maxBytesPerImage }, sessionId: sid });
    }

    const st = sessionState.get(sid) || { lastTexts: [], updatedAt: Date.now() };
    st.updatedAt = Date.now();
    sessionState.set(sid, st);

    const prompt = buildGeneralPrompt({ meta, lastTexts: st.lastTexts });

    const raw = await geminiGenerateTwoImages({ prompt, prevB64, curB64 });
    const parsed = tryParseJsonLoose(raw) || {};

    let commentary = String(parsed.commentary || "").trim();
    const topic = String(parsed.topic || "その他").slice(0, 20);
    const confidence = clamp(parsed.confidence, 0, 1);

    let advantage = undefined;
    if (meta?.enableAdvantage && parsed?.advantage && typeof parsed.advantage.red === "number") {
      const red = clamp(parsed.advantage.red, 0, 1);
      const blue = (typeof parsed.advantage.blue === "number") ? clamp(parsed.advantage.blue, 0, 1) : (1 - red);
      advantage = {
        red,
        blue,
        reason: String(parsed.advantage.reason || "").slice(0, 80),
      };
    }

    // 重複抑制（サーバ側）
    const norm = normalizeForDedup(commentary);
    const lastNorm = normalizeForDedup(st.lastTexts?.[st.lastTexts.length - 1] || "");
    if (norm && lastNorm && (norm === lastNorm || norm.includes(lastNorm) || lastNorm.includes(norm))) {
      // 似すぎなら控えめな一言に
      commentary = "";
    }

    if (commentary) {
      st.lastTexts.push(commentary);
      if (st.lastTexts.length > 12) st.lastTexts.splice(0, st.lastTexts.length - 12);
    }

    res.json({ sessionId: sid, commentary, topic, confidence, advantage });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server error", detail: String(e?.message ?? e) });
  }
});

// ---- VOICEVOX TTS（任意）: ブラウザから呼ぶ用のプロキシ
// 仕様：/audio_query → /synthesis  :contentReference[oaicite:2]{index=2}
app.post("/api/tts/voicevox", async (req, res) => {
  try {
    const { text, speaker = 0 } = req.body ?? {};
    const t = String(text || "").trim();
    if (!t) return res.status(400).json({ error: "text is required" });

    const sp = Number(speaker) | 0;

    const qUrl = `${VOICEVOX_BASE}/audio_query?text=${encodeURIComponent(t)}&speaker=${encodeURIComponent(sp)}`;
    const qRes = await fetch(qUrl, { method: "POST" });
    if (!qRes.ok) {
      const detail = await qRes.text();
      return res.status(502).json({ error: "VOICEVOX audio_query failed", detail });
    }
    const queryJson = await qRes.json();

    const sUrl = `${VOICEVOX_BASE}/synthesis?speaker=${encodeURIComponent(sp)}`;
    const sRes = await fetch(sUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(queryJson),
    });
    if (!sRes.ok) {
      const detail = await sRes.text();
      return res.status(502).json({ error: "VOICEVOX synthesis failed", detail });
    }

    const wav = Buffer.from(await sRes.arrayBuffer());
    res.setHeader("content-type", "audio/wav");
    res.send(wav);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server error", detail: String(e?.message ?? e) });
  }
});

// 話者一覧の中継（任意）: /speakers  :contentReference[oaicite:3]{index=3}
app.get("/api/tts/voicevox/speakers", async (req, res) => {
  try {
    const r = await fetch(`${VOICEVOX_BASE}/speakers`);
    if (!r.ok) return res.status(502).json({ error: "VOICEVOX speakers failed", detail: await r.text() });
    res.json(await r.json());
  } catch (e) {
    res.status(500).json({ error: "server error", detail: String(e?.message ?? e) });
  }
});

app.use(express.static("public"));
const PORT = Number(process.env.PORT || 3000);
app.listen(PORT, () => console.log(`http://localhost:${PORT}`));
