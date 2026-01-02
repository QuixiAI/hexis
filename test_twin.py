#!/usr/bin/env python3
"""
Test script voor digitale twin
"""

import sys
import os

# Test import
try:
    # Voeg huidige directory toe aan path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("🧪 Testing Digital Twin...")
    print("="*60)
    
    # Test basis versie
    print("\n1️⃣  Testing basis import...")
    from digital_twin import DigitalTwin
    print("   ✅ DigitalTwin class imported successfully")
    
    # Maak test instance
    print("\n2️⃣  Creating test instance...")
    twin = DigitalTwin("test_personality.json")
    print(f"   ✅ Twin created: {twin.personality['name']}")
    
    # Test geheugen
    print("\n3️⃣  Testing memory system...")
    twin.remember("Test memory", "test", 0.8)
    print(f"   ✅ Memory stored: {len(twin.memories)} memories")
    
    # Test chat
    print("\n4️⃣  Testing chat function...")
    response = twin.chat("Hoi!")
    print(f"   ✅ Response generated: {response[:50]}...")
    
    # Test stats
    print("\n5️⃣  Testing stats display...")
    twin.show_stats()
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    if os.path.exists("test_personality.json"):
        os.remove("test_personality.json")
    if os.path.exists("twin_memory.json"):
        os.remove("twin_memory.json")
    print("   ✅ Cleanup complete")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🚀 Je digitale twin is klaar voor gebruik!")
    print("\nStart met: ./start_twin.sh")
    print("Of direct: python3 digital_twin.py")
    print("\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
