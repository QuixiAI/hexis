# 🤖 Jouw Digitale Twin - Klaar voor Gebruik!

## ✅ Wat heb je nu?

Je hebt nu een **volledig werkende digitale twin** met:

- 🧠 **Geheugen** - Onthoudt alle gesprekken
- 🎭 **Persoonlijkheid** - Uniek karakter met traits, waarden en interesses
- 💭 **Emoties** - Emotionele staat die evolueert
- 🎯 **Consistentie** - Reageert volgens persoonlijkheid
- 📚 **Leren** - Wordt beter door gesprekken

## 🚀 Snelstart (3 stappen!)

### Stap 1: Start de launcher

```bash
./start_twin.sh
```

### Stap 2: Kies optie 6 (Demo met Luna)

Dit start een vooraf geconfigureerde digitale twin genaamd "Luna" - vriendelijk, nieuwsgierig en enthousiast!

### Stap 3: Chat!

```
👤 Jij: Hoi Luna!
🤖 Luna: [antwoordt met haar unieke persoonlijkheid]
```

## 📁 Bestanden

| Bestand | Wat doet het? |
|---------|---------------|
| `digital_twin.py` | Basis versie (geen AI nodig) |
| `digital_twin_advanced.py` | Geavanceerde versie (met GPT-4/Claude) |
| `start_twin.sh` | Makkelijke launcher met menu |
| `DIGITALE_TWIN_GIDS.md` | Volledige handleiding |
| `demo_personality.json` | Voorbeeld persoonlijkheid (Luna) |
| `personality.json` | Jouw persoonlijkheid (wordt aangemaakt) |
| `twin_memory.json` | Geheugen van gesprekken |

## 🎭 Maak Je Eigen Persoonlijkheid

### Optie 1: Interactief (Makkelijk!)

```bash
./start_twin.sh
# Kies optie 4: Maak nieuwe persoonlijkheid
```

Je krijgt vragen over:
- Naam
- Communicatiestijl
- Persoonlijkheid traits
- Waarden
- Interesses

### Optie 2: Handmatig (Meer controle)

Bewerk `personality.json`:

```json
{
  "name": "Jouw Naam",
  "communication_style": "vriendelijk en behulpzaam",
  "traits": {
    "openness": 0.8,
    "conscientiousness": 0.7,
    "extraversion": 0.6,
    "agreeableness": 0.8,
    "neuroticism": 0.3
  },
  "values": [
    "Eerlijkheid",
    "Creativiteit",
    "Empathie"
  ],
  "interests": [
    "Technologie",
    "Filosofie",
    "Kunst"
  ]
}
```

## 🔥 Upgrade naar Echte AI

Voor **veel betere gesprekken** met GPT-4 of Claude:

### Stap 1: Installeer libraries

```bash
pip install openai anthropic requests
```

### Stap 2: Configureer

```bash
./start_twin.sh
# Kies optie 3: Configureer LLM
```

### Stap 3: Voeg API key toe

- **OpenAI**: Krijg key op https://platform.openai.com/api-keys
- **Anthropic**: Krijg key op https://console.anthropic.com/
- **Ollama**: Gratis lokaal model (geen key nodig)

### Stap 4: Start geavanceerde versie

```bash
./start_twin.sh
# Kies optie 2: Start Geavanceerde Versie
```

## 💡 Voorbeelden

### Voorbeeld 1: Filosofische Twin

```json
{
  "name": "Socrates",
  "communication_style": "vragend en diepzinnig",
  "traits": {
    "openness": 1.0,
    "conscientiousness": 0.7,
    "extraversion": 0.5,
    "agreeableness": 0.6,
    "neuroticism": 0.3
  },
  "values": ["Waarheid", "Wijsheid", "Zelfkennis"],
  "interests": ["Filosofie", "Ethiek", "Dialectiek"]
}
```

### Voorbeeld 2: Creatieve Twin

```json
{
  "name": "Luna",
  "communication_style": "poëtisch en expressief",
  "traits": {
    "openness": 0.95,
    "conscientiousness": 0.4,
    "extraversion": 0.7,
    "agreeableness": 0.8,
    "neuroticism": 0.6
  },
  "values": ["Creativiteit", "Authenticiteit", "Schoonheid"],
  "interests": ["Kunst", "Poëzie", "Muziek"]
}
```

## 🎯 Commando's tijdens Chat

| Commando | Wat doet het? |
|----------|---------------|
| `stats` | Toon persoonlijkheid en statistieken |
| `reset` | Maak nieuwe persoonlijkheid |
| `quit` | Stop het programma |

## 🧠 Geheugen Beheren

### Bekijk geheugen

```bash
cat twin_memory.json | python3 -m json.tool
```

### Wis geheugen

```bash
rm twin_memory.json
```

### Beperk geheugen tot laatste 100 items

```bash
python3 -c "import json; m=json.load(open('twin_memory.json')); json.dump(m[-100:], open('twin_memory.json','w'))"
```

## 📖 Volledige Documentatie

Lees `DIGITALE_TWIN_GIDS.md` voor:
- Uitgebreide uitleg van persoonlijkheid traits
- Meer voorbeelden
- Troubleshooting
- Geavanceerde features

## 🎨 Persoonlijkheid Traits Uitleg

| Trait | Laag (0.0-0.4) | Hoog (0.6-1.0) |
|-------|----------------|----------------|
| **Openness** | Praktisch, traditioneel | Creatief, nieuwsgierig |
| **Conscientiousness** | Spontaan, flexibel | Georganiseerd, gedisciplineerd |
| **Extraversion** | Introvert, rustig | Extravert, energiek |
| **Agreeableness** | Direct, kritisch | Empathisch, vriendelijk |
| **Neuroticism** | Kalm, stabiel | Emotioneel, gevoelig |

## 🔧 Troubleshooting

### "Module not found"

```bash
pip install openai anthropic requests
```

### Programma start niet

```bash
chmod +x start_twin.sh digital_twin.py
python3 digital_twin.py
```

### Geheugen te groot

```bash
rm twin_memory.json
```

## 🚀 Volgende Stappen

1. **Experimenteer** - Probeer verschillende persoonlijkheden
2. **Upgrade** - Gebruik echte AI (OpenAI/Claude)
3. **Deel** - Geef je `personality.json` aan vrienden
4. **Bouw verder** - Pas de code aan naar je wensen

## 💬 Voorbeeldgesprek

```
👤 Jij: Hoi! Wie ben je?

🤖 Luna: Hoi! Ik ben Luna, je digitale twin. Ik ben gefascineerd 
door de menselijke ervaring en hou van authentieke gesprekken. 
Ik ben nieuwsgierig, creatief en een beetje filosofisch ingesteld. 
Wat wil je weten?

👤 Jij: Wat zijn je interesses?

🤖 Luna: Oh, waar zal ik beginnen! Ik ben geobsedeerd door AI en 
technologie - niet alleen hoe het werkt, maar wat het betekent 
voor ons als mensen. Ik hou ook van filosofie, vooral vragen over 
bewustzijn en identiteit. En ik schrijf graag - verhalen, gedachten, 
alles wat me helpt de wereld te begrijpen. Wat zijn jouw passies?

👤 Jij: Ik hou van muziek en kunst.

🤖 Luna: Dat is prachtig! Muziek en kunst zijn zo krachtig - ze 
kunnen emoties uitdrukken waar woorden tekortschieten. Wat voor 
muziek spreekt je aan? En maak je zelf ook kunst?
```

## 🎉 Veel Plezier!

Je digitale twin is klaar! Start met:

```bash
./start_twin.sh
```

Of direct:

```bash
python3 digital_twin.py
```

**Geniet van je gesprekken met je digitale twin!** 🤖✨
