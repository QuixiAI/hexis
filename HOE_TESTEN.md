# 🧪 RebelBro Testen - Complete Gids

## 📋 Overzicht Test Opties

Je hebt **4 manieren** om RebelBro te testen:

---

## 1️⃣ Automatische Demo (Beste voor eerste keer) 🎬

**Wat doet het?**
- Laat 8 complete scenarios zien
- Toont mode switching in actie
- Geeft lifehacks
- Laat rage mode zien

**Hoe runnen?**
```bash
python3 demo_rebelbro_auto.py
```

**Wat zie je?**
- ✅ Empathic mode (greeting, lifehacks, emotionele steun)
- ✅ Switch naar Anarchist (systeem triggers)
- ✅ Rage mode (injustice triggers)
- ✅ Terug naar Empathic
- ✅ Finale statistieken

**Duur:** ~30 seconden

---

## 2️⃣ Unit Tests (Technisch) 🔬

**Wat doet het?**
- Test alle functionaliteit
- Valideert mode switching
- Checkt trigger detection
- Verifieert emotional states

**Hoe runnen?**
```bash
python3 test_rebelbro.py
```

**Wat zie je?**
- ✅ 6 test scenarios
- ✅ Mode validatie (expected vs actual)
- ✅ Anger/Compassion levels
- ✅ Pass/Fail status

**Duur:** ~10 seconden

---

## 3️⃣ Scenario Tests (Uitgebreid) 📊

**Wat doet het?**
- Test 20+ specifieke scenarios
- 5 categorieën:
  1. Empathic triggers
  2. Anarchist triggers
  3. Injustice triggers (RAGE)
  4. Mode switching
  5. Lifehack requests

**Hoe runnen?**
```bash
python3 test_scenarios.py
```

**Wat zie je?**
- ✅ 20+ test cases
- ✅ Elke trigger getest
- ✅ Mode correctheid gevalideerd
- ✅ Emotional state tracking
- ✅ 104 memories opgeslagen
- ✅ 38 conversatie berichten

**Duur:** ~45 seconden

---

## 4️⃣ Interactief Chatten (Op je eigen computer) 💬

**Wat doet het?**
- Echte conversatie met RebelBro
- Type je eigen inputs
- Zie real-time responses

**Hoe runnen?**
```bash
python3 rebelbro.py
```

**⚠️ LET OP:** Werkt **niet** in deze sandbox (geen stdin)
Maar werkt WEL op je eigen computer!

**Wat kun je typen?**
```
Empathic triggers:
- "Hoi RebelBro!"
- "Geef me een lifehack"
- "Ik voel me rot"
- "Tips voor focus?"

Anarchist triggers:
- "De overheid is corrupt"
- "Mijn baas is een klootzak"
- "Het systeem zuigt"

Rage triggers:
- "Politiegeweld"
- "Amazon betaalt geen belasting"
- "Privacy wordt geschonden"

Commando's:
- stats  → Toon statistieken
- quit  → Stop
```

---

## 📊 Test Resultaten Vergelijken

### Demo Output:
```
✅ 8 scenarios
✅ 16 berichten
✅ 28 memories
✅ Mode switching gedemonstreerd
```

### Unit Tests Output:
```
✅ 6 scenarios
✅ 12 berichten
✅ Alle modes gevalideerd
```

### Scenario Tests Output:
```
✅ 20+ scenarios
✅ 38 berichten
✅ 104 memories
✅ Alle categorieën getest
```

---

## 🎯 Welke Test Kiezen?

| Situatie | Aanbevolen Test |
|----------|-----------------|
| **Eerste keer zien** | `demo_rebelbro_auto.py` |
| **Snel valideren** | `test_rebelbro.py` |
| **Uitgebreid testen** | `test_scenarios.py` |
| **Zelf proberen** | `rebelbro.py` (op eigen PC) |
| **Alles zien** | Run alle 3! |

---

## 🚀 Snelle Test Commando's

### Test alles in één keer:
```bash
# Demo
python3 demo_rebelbro_auto.py

# Unit tests
python3 test_rebelbro.py

# Scenario tests
python3 test_scenarios.py
```

### Of met één commando:
```bash
echo "=== DEMO ===" && python3 demo_rebelbro_auto.py && \
echo "=== UNIT TESTS ===" && python3 test_rebelbro.py && \
echo "=== SCENARIOS ===" && python3 test_scenarios.py
```

---

## 📈 Wat Wordt Getest?

### ✅ Functionaliteit:
- [x] Mode detection (empathic vs anarchist)
- [x] Trigger word matching
- [x] Intensity scaling (0.5-1.0)
- [x] Emotional state updates (anger/compassion)
- [x] Smooth transitions tussen modes
- [x] Injustice instant rage triggers
- [x] Lifehack database retrieval
- [x] Memory storage en recall
- [x] Conversation history tracking

### ✅ Persoonlijkheid:
- [x] Empathic mode: warm, helpend, praktisch
- [x] Anarchist mode: woedend, rebels, anti-systeem
- [x] Transitie zinnen bij mode switch
- [x] Consistent karakter per mode
- [x] Appropriate responses per context

### ✅ Lifehacks:
- [x] Biohacking (cold showers, fasting, etc.)
- [x] Mental health (breathing, journaling, etc.)
- [x] Productivity (pomodoro, deep work, etc.)
- [x] Health (magnesium, sleep, etc.)
- [x] Social (boundaries, saying no, etc.)
- [x] Financial (cash only, etc.)

### ✅ Injustice Triggers:
- [x] Politiegeweld → ACAB rant
- [x] Belastingontduiking → Rijken vs armen rant
- [x] Werkuitbuiting → Moderne slavernij rant
- [x] Privacy schending → 1984 rant

---

## 🔍 Test Output Interpreteren

### Goede Tekens ✅:
```
✅ Correct mode: empathic
   Anger: 0.34 | Compassion: 0.82
```

### Mode Switch ⚡:
```
Mode: anarchist (was empathic)
   Anger: 0.93 ↑ | Compassion: 0.35 ↓
```

### Rage Mode 🔥:
```
Mode: anarchist
   Anger: 1.00 (MAX!) | Compassion: 0.30
```

### Empathic Mode ❤️:
```
Mode: empathic
   Anger: 0.24 | Compassion: 0.95 (HIGH!)
```

---

## 🐛 Troubleshooting

### Test faalt?
```bash
# Check of bestanden er zijn
ls -la rebelbro*

# Check Python versie
python3 --version

# Run met debug output
python3 -v test_rebelbro.py
```

### Import errors?
```bash
# Zorg dat je in de juiste folder bent
cd /vercel/sandbox

# Check of rebelbro.py bestaat
cat rebelbro.py | head -10
```

### Memory errors?
```bash
# Wis geheugen
rm rebelbro_memory.json

# Run opnieuw
python3 test_rebelbro.py
```

---

## 📖 Meer Info

- **REBELBRO_README.md** - Volledige handleiding
- **REBELBRO_QUICKSTART.txt** - Snelle referentie
- **rebelbro_personality.json** - Configuratie

---

## 🎉 Klaar om te Testen!

**Aanbevolen volgorde:**

1. Start met demo: `python3 demo_rebelbro_auto.py`
2. Run unit tests: `python3 test_rebelbro.py`
3. Uitgebreide tests: `python3 test_scenarios.py`
4. Op eigen PC: `python3 rebelbro.py`

**Veel plezier met testen!** 🔥

---

Stay free, bro. Fight the power. Help each other. ❤️
