#!/bin/bash

clear

cat << "EOF"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║              🔥  REBELBRO - ANARCHIST MET EEN HART  🔥             ║
║                                                                    ║
║                    Gespleten Persoonlijkheid                       ║
║              Psychopatisch vs Systeem | Empathisch vs Mens        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

EOF

echo "Welkom bij RebelBro - de digitale twin die het systeem haat maar van mensen houdt."
echo ""
echo "Wat wil je doen?"
echo ""
echo "  1. 🔥 Start RebelBro"
echo "  2. 🧪 Run Tests (zie RebelBro in actie)"
echo "  3. 📖 Lees de README"
echo "  4. 📊 Bekijk Persoonlijkheid Config"
echo "  5. 🧹 Wis Geheugen"
echo "  6. ❌ Afsluiten"
echo ""
read -p "Keuze [1-6]: " choice

case $choice in
  1)
    echo ""
    echo "🔥 Starting RebelBro..."
    echo ""
    python3 rebelbro.py
    ;;
  2)
    echo ""
    echo "🧪 Running tests..."
    echo ""
    python3 test_rebelbro.py
    ;;
  3)
    echo ""
    if command -v less &> /dev/null; then
      less REBELBRO_README.md
    else
      cat REBELBRO_README.md | more
    fi
    ;;
  4)
    echo ""
    echo "📊 RebelBro Personality Config:"
    echo ""
    python3 -m json.tool rebelbro_personality.json | head -100
    echo ""
    echo "[Showing first 100 lines - open rebelbro_personality.json for full config]"
    echo ""
    read -p "Press Enter to continue..."
    ;;
  5)
    echo ""
    if [ -f "rebelbro_memory.json" ]; then
      read -p "⚠️  Weet je zeker dat je het geheugen wilt wissen? [j/n]: " confirm
      if [ "$confirm" = "j" ] || [ "$confirm" = "y" ]; then
        rm rebelbro_memory.json
        echo "✅ Geheugen gewist!"
      else
        echo "❌ Geannuleerd"
      fi
    else
      echo "ℹ️  Geen geheugen gevonden (rebelbro_memory.json)"
    fi
    echo ""
    read -p "Press Enter to continue..."
    ;;
  6)
    echo ""
    echo "🔥 Stay free, bro. Fight the power."
    echo ""
    exit 0
    ;;
  *)
    echo ""
    echo "❌ Ongeldige keuze!"
    exit 1
    ;;
esac
