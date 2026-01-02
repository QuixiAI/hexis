#!/usr/bin/env python3
"""
RebelBro Automatische Demo
Laat RebelBro zien in actie met verschillende scenarios
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rebelbro import RebelBro

def print_scenario(number, title):
    print(f"\n{'─'*70}")
    print(f"  SCENARIO {number}: {title}")
    print(f"{'─'*70}\n")

def simulate_conversation(rebelbro, user_input):
    """Simuleer een gesprek"""
    print(f"👤 Jij: {user_input}")
    
    response = rebelbro.chat(user_input)
    
    print(f"\n🔥 RebelBro [{rebelbro.current_mode.upper()}]:")
    # Toon eerste 300 karakters voor leesbaarheid
    if len(response) > 300:
        print(f"{response[:300]}...")
    else:
        print(response)
    
    print(f"\n   💭 Emotional State: Anger={rebelbro.anger_level:.2f} | Compassion={rebelbro.compassion_level:.2f}")

def main():
    print("\n" + "="*70)
    print("🔥 REBELBRO - AUTOMATISCHE DEMO")
    print("="*70)
    print("\nDit is een demo van RebelBro's gespleten persoonlijkheid.")
    print("Let op hoe hij schakelt tussen Anarchist en Empathic mode!\n")
    
    # Initialize RebelBro
    print("🔧 Initializing RebelBro...")
    rebelbro = RebelBro()
    print("✅ RebelBro is online!\n")
    
    # Scenario 1: Greeting (Empathic)
    print_scenario(1, "Eerste Contact (Empathic Mode)")
    simulate_conversation(rebelbro, "Hoi RebelBro! Wie ben je?")
    
    # Scenario 2: Lifehack Request
    print_scenario(2, "Vraag om Lifehack (Blijft Empathic)")
    simulate_conversation(rebelbro, "Geef me een tip voor betere focus")
    
    # Scenario 3: Emotional Support
    print_scenario(3, "Emotionele Steun (Deep Empathic)")
    simulate_conversation(rebelbro, "Ik voel me zo moe en uitgeput, alles is zwaar")
    
    # Scenario 4: System Trigger (Switch to Anarchist)
    print_scenario(4, "Systeem Trigger (SWITCH → Anarchist Mode)")
    simulate_conversation(rebelbro, "De overheid verhoogt weer de belasting terwijl de rijken niks betalen")
    
    # Scenario 5: Work Exploitation
    print_scenario(5, "Werkuitbuiting (Blijft Anarchist)")
    simulate_conversation(rebelbro, "Mijn baas wil dat ik gratis overuren maak")
    
    # Scenario 6: Injustice Trigger (RAGE MODE)
    print_scenario(6, "Onrecht Trigger (INSTANT RAGE MODE)")
    simulate_conversation(rebelbro, "Ik hoorde over politiegeweld in de stad")
    
    # Scenario 7: Back to Empathic
    print_scenario(7, "Terug naar Empathie (Switch → Empathic)")
    simulate_conversation(rebelbro, "Bedankt voor je steun bro, je helpt me echt")
    
    # Scenario 8: Another Lifehack
    print_scenario(8, "Nog een Lifehack (Blijft Empathic)")
    simulate_conversation(rebelbro, "Heb je tips voor betere slaap?")
    
    # Final Stats
    print("\n" + "="*70)
    print("📊 FINALE STATISTIEKEN:")
    print("="*70)
    rebelbro.show_stats()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLEET!")
    print("="*70)
    print("\n🔥 Zoals je ziet:")
    print("   ✓ RebelBro schakelt automatisch tussen modes")
    print("   ✓ Empathic mode: warm, helpend, praktisch")
    print("   ✓ Anarchist mode: woedend, rebels, anti-systeem")
    print("   ✓ Injustice triggers: instant RAGE")
    print("   ✓ Smooth transitions tussen modes")
    print("   ✓ Onthoudt alles in geheugen")
    
    print(f"\n📚 Totaal gesprekken: {len(rebelbro.conversation_history)} berichten")
    print(f"💾 Totaal herinneringen: {len(rebelbro.memories)}")
    
    print("\n💡 Wil je zelf met RebelBro praten?")
    print("   Run: python3 rebelbro.py")
    
    print("\n📖 Documentatie:")
    print("   • REBELBRO_QUICKSTART.txt - Snelle referentie")
    print("   • REBELBRO_README.md - Volledige handleiding")
    print("   • rebelbro_personality.json - Configuratie")
    
    print("\n🔥 Stay free, bro. Fight the power. Help each other. ❤️\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
