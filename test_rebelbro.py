#!/usr/bin/env python3
"""
Test RebelBro met verschillende scenarios
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rebelbro import RebelBro

def test_rebelbro():
    print("🧪 Testing RebelBro...")
    print("="*70)
    
    # Initialize
    print("\n1️⃣  Initializing RebelBro...")
    rebelbro = RebelBro()
    print(f"   ✅ RebelBro initialized: {rebelbro.personality['name']}")
    print(f"   Current mode: {rebelbro.current_mode}")
    
    # Test scenarios
    scenarios = [
        {
            "name": "Greeting (Empathic)",
            "input": "Hoi RebelBro!",
            "expected_mode": "empathic"
        },
        {
            "name": "Request for lifehack",
            "input": "Geef me een lifehack voor betere focus",
            "expected_mode": "empathic"
        },
        {
            "name": "Emotional support",
            "input": "Ik voel me zo moe en uitgeput",
            "expected_mode": "empathic"
        },
        {
            "name": "System trigger (Anarchist)",
            "input": "De overheid neemt weer meer belasting",
            "expected_mode": "anarchist"
        },
        {
            "name": "Injustice trigger (RAGE)",
            "input": "Ik hoorde over politiegeweld in de stad",
            "expected_mode": "anarchist"
        },
        {
            "name": "Back to empathic",
            "input": "Bedankt voor je hulp bro",
            "expected_mode": "empathic"
        }
    ]
    
    print("\n2️⃣  Testing scenarios...\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─'*70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'─'*70}")
        print(f"👤 Input: {scenario['input']}")
        
        response = rebelbro.chat(scenario['input'])
        
        print(f"\n🔥 RebelBro [{rebelbro.current_mode.upper()}]:")
        print(f"{response[:200]}..." if len(response) > 200 else response)
        
        mode_match = rebelbro.current_mode == scenario['expected_mode']
        status = "✅" if mode_match else "⚠️"
        print(f"\n{status} Mode: {rebelbro.current_mode} (expected: {scenario['expected_mode']})")
        print(f"   Anger: {rebelbro.anger_level:.2f} | Compassion: {rebelbro.compassion_level:.2f}")
    
    # Show final stats
    print(f"\n\n{'='*70}")
    print("3️⃣  Final Stats:")
    print(f"{'='*70}")
    rebelbro.show_stats()
    
    print("\n✅ ALL TESTS COMPLETED!")
    print("\n🔥 RebelBro is ready to fight the system and help people!")
    print("\nStart met: python3 rebelbro.py")
    print("\n")

if __name__ == "__main__":
    try:
        test_rebelbro()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
