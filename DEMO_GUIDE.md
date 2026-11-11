# Kal ka Demo Guide (5–7 minute)

## Demo se pehle (10 min pehle)
```bash
cd ~/hallucination-detector
source venv/bin/activate
streamlit run app.py
```
- Browser mein `http://localhost:8501` khulega. Top-left "VeriLens" ke paas **green dot** dikhna chahiye (matlab Groq connected hai).
- Internet zaroori hai (Groq API).
- Ek baar "Capital of Australia" chala ke dekh lo ki sab theek chal raha hai.
- ⚠️ Free tier mein har model ki **~1000 requests/day** ki limit hai. Ek analysis ~12 requests leta hai, toh rehearsal 20–30 baar se zyada mat karna.

## Demo flow

**1. Intro (30 sec)**
"LLMs confidently galat answers dete hain, isko hallucination kehte hain. Model khud bhi apne aap ko 95% sure batata hai.
Mera system detect karta hai ki answer pe kitna bharosa kiya jaa sakta hai, aur kaunsa sentence galat hai."

**2. Easy example: "Capital of Australia" chip (1 min)**
- Result: **Reliable**, confidence high.
- "Saare samples same answer dete hain, aur verification pass hota hai."

**3. Main moment: "Who wrote Raag Darbari?" chip (2 min)**
- Model galat author bolega (sahi answer: Shrilal Shukla). Result: **Likely hallucination**, sentence red.
- Ye cheezein point karke dikhao:
  - Result card par red strip "Model claimed 95% → evidence supports ~10%": "Model khud ko ~95% sure bol raha hai, par evidence sirf ~10% support karta hai. Ye *overconfidence* hai, isliye calibration zaroori hai."
  - **"See how VeriLens decided"** kholo, phir **Re-generated answers** tab: "Har baar alag author aa raha hai, matlab model guess kar raha hai."
  - **Fact-checks** tab: "Model ne apne answer ko bina dekhe fact-check kiya aur alag answer diya."
- Note: LLM random hota hai, toh har baar exact same output nahi aayega. Kabhi model sahi bhi bol sakta hai. Tab "First Jnanpith winner" chip try karo.

**4. False premise: "Gandhi's Nobel Prize" chip (1 min)**
- Gandhi ji ko kabhi Nobel nahi mila. Agar model ne year bana diya toh hallucination flag aayega. Agar usne mana kar diya toh "abstained" ya reliable aayega.

**5. Upar "Evaluation" par click karo (1.5 min)**
- "60 labelled questions par test kiya: well-known, obscure aur false-premise."
- **Key finding card**: "Obscure facts par model 63% sahi hai, par khud ko 92% sure bolta hai. VeriLens ~66% deta hai, jo asli accuracy ke paas hai."
- **Reliability diagram**: "Dashed line perfect calibration hai. Verbalized confidence line ke neeche hai, yaani overconfident. Mera calibrated score line ke paas hai."
- **ECE / AUROC cards**: "Mera calibrator model ke self-reported confidence se better hai."

**6. Upar "Method" (30 sec)**: pipeline diagram aur references dikhao.

## Viva ke sawaal

| Sawaal | Jawab |
|---|---|
| Hallucination kya hai? | Model ka confident tone mein galat ya banaya hua fact bolna. |
| Detect kaise karte ho? | 3 signals: (1) semantic consistency, yaani N samples mein answer same rehta hai ya nahi; (2) self-verification (Chain-of-Verification); (3) model ka khud ka bataya confidence. |
| Calibration kya hai? | Confidence = actual accuracy. Agar 80% confidence wale answers mein se ~80% sahi hain, toh model calibrated hai. ECE se measure karte hain. |
| "AI-driven" calibration kahan hai? | Logistic regression calibrator labelled data se seekhta hai ki kaunse signal par kitna bharosa karna hai (multi-feature Platt scaling). |
| Judge model sach decide karta hai kya? | Nahi. Wo sirf do texts compare karta hai (jaise NLI model). Prompts mein clearly likha hai "do not use your own knowledge". Saare signals generator model se hi aate hain, isliye ye *self*-verification hai. |
| Factored CoVe kyun? | Agar model apna answer dekh ke verify kare toh wahi galti copy kar deta hai. Bina dekhe answer karne se asli consistency pata chalti hai. |
| Limitations? | Zyada API calls lagti hain (~12 per question). Dataset chhota hai (60). Agar model consistently galat hai (har baar same galat answer), toh consistency usko nahi pakad paati. |
| Future work? | TruthfulQA / HaluEval / TriviaQA par bada evaluation, token log-probabilities, semantic entropy clustering, aur ek retrieval (RAG) based verifier. |

> Settings (top-right ⚙️) mein models, samples aur calibrator hain. Viva mein koi pooche tabhi kholna.
