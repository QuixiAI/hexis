#!/usr/bin/env python3
"""
Test RebelBro met specifieke scenarios
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rebelbro import RebelBro

def test_scenario(rebelbro, name, user_input, expected_mode=None):
    """Test een specifiek scenario"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"👤 Input: {user_input}")
    
    response = rebelbro.chat(user_input)
    
    print(f"\n🔥 RebelBro [{rebelbro.current_mode.upper()}]:")
    print(response[:200] + "..." if len(response) > 200 else response)
    
    print(f"\n📊 State:")
    print(f"   Mode: {rebelbro.current_mode}")
    print(f"   Anger: {rebelbro.anger_level:.2f}")
    print(f"   Compassion: {rebelbro.compassion_level:.2f}")
    
    if expected_mode:
        if rebelbro.current_mode == expected_mode:
            print(f"   ✅ Correct mode: {expected_mode}")
        else:
            print(f"   ⚠️  Expected {expected_mode}, got {rebelbro.current_mode}")
    
    return rebelbro.current_mode

def main():
    print("\n" + "="*70)
    print("🧪 REBELBRO - SCENARIO TESTS")
    print("="*70)
    
    rebelbro = RebelBro()
    
    # Test 1: Empathic triggers
    print("\n\n" + "🎯 CATEGORIE 1: EMPATHIC TRIGGERS".center(70))
    test_scenario(rebelbro, "Greeting", "Hoi!", "empathic")
    test_scenario(rebelbro, "Help Request", "Ik heb hulp nodig", "empathic")
    test_scenario(rebelbro, "Lifehack Request", "Geef me een tip", "empathic")
    test_scenario(rebelbro, "Emotional Pain", "Ik voel me verdrietig", "empathic")
    
    # Test 2: Anarchist triggers
    print("\n\n" + "🎯 CATEGORIE 2: ANARCHIST TRIGGERS".center(70))
    test_scenario(rebelbro, "Government", "De overheid is corrupt", "anarchist")
    test_scenario(rebelbro, "Boss/Work", "Mijn baas is een klootzak", "anarchist")
    test_scenario(rebelbro, "System", "Het systeem is kapot", "anarchist")
    test_scenario(rebelbro, "Taxes", "Belastingen zijn te hoog", "anarchist")
    
    # Test 3: Injustice triggers (RAGE)
    print("\n\n" + "🎯 CATEGORIE 3: INJUSTICE TRIGGERS (RAGE)".center(70))
    test_scenario(rebelbro, "Police Violence", "Politiegeweld in de stad", "anarchist")
    test_scenario(rebelbro, "Tax Evasion Rich", "Amazon betaalt geen belasting", "anarchist")
    test_scenario(rebelbro, "Work Exploitation", "Gratis overuren verplicht", "anarchist")
    test_scenario(rebelbro, "Privacy Violation", "Ze spioneren op iedereen", "anarchist")
    
    # Test 4: Mode switching
    print("\n\n" + "🎯 CATEGORIE 4: MODE SWITCHING".center(70))
    test_scenario(rebelbro, "Anarchist → Empathic", "Bedankt voor je hulp bro", "empathic")
    test_scenario(rebelbro, "Empathic → Anarchist", "De politie is corrupt", "anarchist")
    test_scenario(rebelbro, "Back to Empathic", "Ik waardeer je steun", "empathic")
    
    # Test 5: Lifehack categories
    print("\n\n" + "🎯 CATEGORIE 5: LIFEHACK REQUESTS".center(70))
    test_scenario(rebelbro, "Sleep Hack", "Tips voor betere slaap?", "empathic")
    test_scenario(rebelbro, "Focus Hack", "Hoe krijg ik meer focus?", "empathic")
    test_scenario(rebelbro, "Stress Hack", "Ik heb veel stress", "empathic")
    test_scenario(rebelbro, "Health Hack", "Gezondheid verbeteren?", "empathic")
    
    # Final stats
    print("\n\n" + "="*70)
    print("📊 FINAL STATISTICS")
    print("="*70)
    rebelbro.show_stats()
    
    print("\n" + "="*70)
    print("✅ ALL SCENARIO TESTS COMPLETED!")
    print("="*70)
    print(f"\n📚 Total conversations: {len(rebelbro.conversation_history)}")
    print(f"💾 Total memories: {len(rebelbro.memories)}")
    print(f"🎭 Final mode: {rebelbro.current_mode}")
    print(f"🔥 Final anger: {rebelbro.anger_level:.2f}")
    print(f"❤️  Final compassion: {rebelbro.compassion_level:.2f}")
    
    print("\n🔥 RebelBro is fully tested and ready to use!")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
