#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║           🤖  DIGITALE TWIN - LAUNCHER  🤖                 ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Welkom! Wat wil je doen?"
echo ""
echo "  1. 🚀 Start Basis Versie (geen AI, voor testen)"
echo "  2. 🔥 Start Geavanceerde Versie (met echte AI)"
echo "  3. ⚙️  Configureer LLM (OpenAI, Claude, Ollama)"
echo "  4. 🎭 Maak nieuwe persoonlijkheid"
echo "  5. 📖 Lees de gids"
echo "  6. 🧪 Demo met Luna (vooraf geconfigureerd)"
echo "  7. ❌ Afsluiten"
echo ""
read -p "Keuze [1-7]: " choice

case $choice in
  1)
    echo ""
    echo "🚀 Start basis versie..."
    python3 digital_twin.py
    ;;
  2)
    echo ""
    echo "🔥 Start geavanceerde versie..."
    python3 digital_twin_advanced.py
    ;;
  3)
    echo ""
    echo "⚙️  LLM configuratie..."
    python3 digital_twin_advanced.py configure
    ;;
  4)
    echo ""
    echo "🎭 Nieuwe persoonlijkheid maken..."
    rm -f personality.json
    python3 digital_twin.py
    ;;
  5)
    echo ""
    if command -v less &> /dev/null; then
      less DIGITALE_TWIN_GIDS.md
    else
      cat DIGITALE_TWIN_GIDS.md
    fi
    ;;
  6)
    echo ""
    echo "🧪 Demo met Luna..."
    cp demo_personality.json personality.json
    echo "✅ Luna persoonlijkheid geladen!"
    echo ""
    python3 digital_twin.py
    ;;
  7)
    echo ""
    echo "👋 Tot ziens!"
    exit 0
    ;;
  *)
    echo ""
    echo "❌ Ongeldige keuze!"
    exit 1
    ;;
esac
