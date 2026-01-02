#!/usr/bin/env python3
"""
RebelBro Interactive Demo
Laat RebelBro zien in actie met verschillende scenarios
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rebelbro import RebelBro

def print_separator():
    print("\n" + "="*70 + "\n")

def print_scenario(number, title):
    print(f"\n{'─'*70}")
    print(f"  SCENARIO {number}: {title}")
    print(f"{'─'*70}\n")

def simulate_conversation(rebelbro, user_input, pause=2):
    """Simuleer een gesprek met pauzes"""
    print(f"👤 Jij: {user_input}")
    time.sleep(0.5)
    
    response = rebelbro.chat(user_input)
    
    print(f"\n🔥 RebelBro [{rebelbro.current_mode.upper()}]:")
    print(f"{response}")
    
    print(f"\n   💭 Anger: {rebelbro.anger_level:.2f} | Compassion: {rebelbro.compassion_level:.2f}")
    
    time.sleep(pause)

def main():
    print("\n" + "="*70)
    print("🔥 REBELBRO - INTERACTIEVE DEMO")
    print("="*70)
    print("\nDit is een demo van RebelBro's gespleten persoonlijkheid.")
    print("Let op hoe hij schakelt tussen Anarchist en Empathic mode!\n")
    
    input("Druk op Enter om te starten...")
    
    # Initialize RebelBro
    print("\n🔧 Initializing RebelBro...")
    rebelbro = RebelBro()
    print("✅ RebelBro is online!\n")
    
    time.sleep(1)
    
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
    
    # Scenario 5: Injustice Trigger (RAGE MODE)
    print_scenario(5, "Onrecht Trigger (INSTANT RAGE MODE)")
    simulate_conversation(rebelbro, "Ik hoorde over politiegeweld in de stad, ze sloegen een ongewapende man")
    
    # Scenario 6: Back to Empathic
    print_scenario(6, "Terug naar Empathie (Switch → Empathic)")
    simulate_conversation(rebelbro, "Bedankt voor je steun bro, je helpt me echt")
    
    # Scenario 7: Another Lifehack
    print_scenario(7, "Nog een Lifehack (Blijft Empathic)")
    simulate_conversation(rebelbro, "Heb je tips voor betere slaap?")
    
    # Final Stats
    print_separator()
    print("📊 FINALE STATISTIEKEN:")
    print_separator()
    rebelbro.show_stats()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLEET!")
    print("="*70)
    print("\n🔥 Zoals je ziet:")
    print("   • RebelBro schakelt automatisch tussen modes")
    print("   • Empathic mode: warm, helpend, praktisch")
    print("   • Anarchist mode: woedend, rebels, anti-systeem")
    print("   • Injustice triggers: instant RAGE")
    print("   • Smooth transitions tussen modes")
    print("   • Onthoudt alles in geheugen")
    
    print("\n💡 Wil je zelf met RebelBro praten?")
    print("   Run: python3 rebelbro.py")
    
    print("\n🔥 Stay free, bro. Fight the power. Help each other. ❤️\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🔥 Demo gestopt. Peace out, bro!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
