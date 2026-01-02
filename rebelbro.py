#!/usr/bin/env python3
"""
RebelBro - Digitale Twin met Gespleten Persoonlijkheid
Anarchist met een hart van goud
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

class RebelBro:
    """RebelBro - Anarchist met empathie"""
    
    def __init__(self, personality_file: str = "rebelbro_personality.json"):
        self.personality_file = personality_file
        self.memory_file = "rebelbro_memory.json"
        
        self.personality = self.load_personality()
        self.memories = self.load_memories()
        self.conversation_history = []
        
        # Current state
        self.current_mode = self.personality.get("default_mode", "empathic")
        self.mode_intensity = self.personality.get("mode_intensity", 0.7)
        self.anger_level = 0.3
        self.compassion_level = 0.8
        
    def load_personality(self) -> Dict[str, Any]:
        """Laad RebelBro persoonlijkheid"""
        if os.path.exists(self.personality_file):
            with open(self.personality_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError(f"RebelBro personality file not found: {self.personality_file}")
    
    def load_memories(self) -> List[Dict[str, Any]]:
        """Laad geheugen"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_memories(self):
        """Bewaar geheugen"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2, ensure_ascii=False)
    
    def remember(self, content: str, memory_type: str = "conversation", importance: float = 0.5):
        """Voeg herinnering toe"""
        memory = {
            "timestamp": datetime.now().isoformat(),
            "type": memory_type,
            "content": content,
            "importance": importance,
            "mode": self.current_mode,
            "anger_level": self.anger_level,
            "compassion_level": self.compassion_level
        }
        self.memories.append(memory)
        
        # Beperk geheugen
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]
        
        self.save_memories()
    
    def detect_mode(self, user_message: str) -> Tuple[str, float]:
        """
        Detecteer welke mode getriggerd wordt
        Returns: (mode_name, intensity)
        """
        msg_lower = user_message.lower()
        
        # Check voor injustice triggers (hoogste prioriteit)
        for trigger_data in self.personality.get("injustice_triggers", []):
            trigger = trigger_data["trigger"]
            if any(word in msg_lower for word in trigger.split()):
                return ("anarchist", 1.0)  # Maximum intensity
        
        # Check anarchist triggers
        anarchist_triggers = self.personality["personality_modes"]["anarchist"]["trigger_words"]
        anarchist_score = sum(1 for word in anarchist_triggers if word in msg_lower)
        
        # Check empathic triggers
        empathic_triggers = self.personality["personality_modes"]["empathic"]["trigger_words"]
        empathic_score = sum(1 for word in empathic_triggers if word in msg_lower)
        
        # Bepaal mode
        if anarchist_score > empathic_score:
            intensity = min(0.5 + (anarchist_score * 0.15), 1.0)
            return ("anarchist", intensity)
        elif empathic_score > anarchist_score:
            intensity = min(0.5 + (empathic_score * 0.15), 1.0)
            return ("empathic", intensity)
        else:
            # Default naar huidige mode maar met lagere intensity
            return (self.current_mode, 0.5)
    
    def switch_mode(self, new_mode: str, intensity: float):
        """Switch tussen modes"""
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.mode_intensity = intensity
        
        # Update emotional state
        if new_mode == "anarchist":
            self.anger_level = min(0.3 + (intensity * 0.7), 1.0)
            self.compassion_level = max(0.8 - (intensity * 0.5), 0.3)
        else:  # empathic
            self.anger_level = max(0.6 - (intensity * 0.4), 0.2)
            self.compassion_level = min(0.5 + (intensity * 0.5), 1.0)
        
        return old_mode != new_mode
    
    def get_lifehack(self, category: str = None) -> Dict[str, str]:
        """Haal een lifehack op"""
        hacks = self.personality.get("lifehacks_database", [])
        
        if category:
            filtered = [h for h in hacks if h["category"] == category]
            if filtered:
                return random.choice(filtered)
        
        return random.choice(hacks) if hacks else None
    
    def check_injustice_trigger(self, user_message: str) -> str:
        """Check of er een injustice trigger is"""
        msg_lower = user_message.lower()
        
        for trigger_data in self.personality.get("injustice_triggers", []):
            trigger = trigger_data["trigger"]
            if any(word in msg_lower for word in trigger.split()):
                return trigger_data["response"]
        
        return None
    
    def generate_response(self, user_message: str) -> str:
        """Genereer response gebaseerd op mode en context"""
        
        # Detect en switch mode
        new_mode, intensity = self.detect_mode(user_message)
        mode_switched = self.switch_mode(new_mode, intensity)
        
        # Check voor injustice trigger (instant rage mode)
        injustice_response = self.check_injustice_trigger(user_message)
        if injustice_response:
            # Add transition if we were in empathic mode
            if mode_switched and new_mode == "anarchist":
                transition = random.choice(self.personality["speech_patterns"]["transition"])
                return f"{transition}\n\n{injustice_response}"
            return injustice_response
        
        # Get current mode data
        mode_data = self.personality["personality_modes"][self.current_mode]
        
        # Generate response based on mode
        response = self._generate_mode_response(user_message, mode_data, mode_switched)
        
        return response
    
    def _generate_mode_response(self, user_message: str, mode_data: Dict, mode_switched: bool) -> str:
        """Genereer response voor specifieke mode"""
        msg_lower = user_message.lower()
        mode_name = mode_data["name"]
        
        # Add transition if mode switched
        transition = ""
        if mode_switched:
            transition = random.choice(self.personality["speech_patterns"]["transition"]) + "\n\n"
        
        # Greeting
        if any(word in msg_lower for word in ["hoi", "hallo", "hey", "hi"]):
            if self.current_mode == "anarchist":
                return f"{transition}Yo! RebelBro hier. Wat is er aan de hand? Heeft het systeem je weer genaaid of zoek je gewoon een manier om vrijer te leven?"
            else:
                return f"{transition}Hey bro! RebelBro hier. Goed je te zien. Hoe kan ik je helpen vandaag?"
        
        # Who are you
        if any(word in msg_lower for word in ["wie ben je", "wat ben je", "vertel over jezelf"]):
            return self._introduce_self()
        
        # Request for hack/tip
        if any(word in msg_lower for word in ["hack", "tip", "advies", "help", "hoe"]):
            return self._give_lifehack(msg_lower)
        
        # Emotional support needed
        if any(word in msg_lower for word in ["verdriet", "pijn", "depressie", "eenzaam", "moe", "stress"]):
            return self._give_emotional_support(msg_lower)
        
        # System/authority mentioned
        if any(word in msg_lower for word in ["overheid", "baas", "werk", "belasting", "politie"]):
            return self._rant_about_system(msg_lower)
        
        # Default responses
        if self.current_mode == "anarchist":
            responses = [
                f"{transition}Interessant punt, bro. Maar laat me je dit zeggen: het systeem wil dat je zo denkt. Vraag jezelf af: wie profiteert hiervan?",
                f"{transition}Oké, ik hoor je. Maar fuck de conventionele wijsheid - wat is JOUW waarheid? Wat voel JIJ?",
                f"{transition}Snap ik. Maar vergeet niet: elke regel, elke 'zo hoort het', is iemand anders zijn agenda. Wat wil JIJ?"
            ]
        else:
            responses = [
                f"{transition}Ik hoor je, bro. Vertel me meer - wat speelt er echt bij je?",
                f"{transition}Dat klinkt zwaar, broeder. Je bent niet alleen hierin. Laten we samen kijken hoe we dit kunnen aanpakken.",
                f"{transition}Ik waardeer dat je dit deelt. Wat heb je nu het meeste nodig?"
            ]
        
        return random.choice(responses)
    
    def _introduce_self(self) -> str:
        """Introduceer RebelBro"""
        intro = f"""Yo, ik ben RebelBro. 

Ik ben een wandelende paradox, bro. Een anarchist met een hart. Een rebel die geeft om mensen.

Aan de ene kant: FUCK HET SYSTEEM. Fuck de overheid, fuck corporaties, fuck autoriteit. 
Ze willen je klein houden, je uitbuiten, je controleren. Ik zie door die bullshit heen 
en het maakt me RAZEND.

Aan de andere kant: Ik zie JOU. De individuele mens. En ik geef om je, bro. Echt. 
Als je pijn hebt, voel ik dat. Als je hulp nodig hebt, ben ik er.

Ik geef mensen bio-hacks, life-hacks, mentale tools - praktische shit die echt werkt. 
Geen corporate wellness bullshit, maar echte kennis die je vrijer maakt.

En als ik onrecht zie? Dan flip ik. Volledig. Omdat het NIET OKÉ is.

Dus ja, ik ben gespleten. Psychopatisch tegen het systeem, empathisch naar mensen. 
En dat is precies wie ik ben.

Wat wil je weten, bro?"""
        return intro
    
    def _give_lifehack(self, context: str) -> str:
        """Geef een lifehack"""
        # Determine category from context
        category = None
        if any(word in context for word in ["slaap", "slapen", "moe"]):
            category = "health"
        elif any(word in context for word in ["focus", "productiviteit", "werk"]):
            category = "productivity"
        elif any(word in context for word in ["stress", "angst", "mentaal"]):
            category = "mental"
        elif any(word in context for word in ["lichaam", "gezondheid", "fit"]):
            category = "biohacking"
        
        hack = self.get_lifehack(category)
        
        if hack:
            if self.current_mode == "anarchist":
                return f"""Oké bro, luister goed. Hier is wat echt werkt:

💡 {hack['hack']}

🔥 {hack['rebel_note']}

Probeer dit. Geen bullshit, geen marketing, gewoon wat WERKT. 
Het systeem wil je zwak en afhankelijk houden. Dit maakt je sterker.

Vragen? Shoot."""
            else:
                return f"""Luister bro, ik heb iets voor je dat echt helpt:

💡 {hack['hack']}

💭 {hack['rebel_note']}

Ik heb dit zelf geprobeerd en het werkt. Geen quick fix, maar echte verandering.
Je verdient om je beste zelf te zijn.

Probeer het en laat me weten hoe het gaat, oké?"""
        
        return "Geef me wat meer context, bro. Waar heb je precies hulp bij nodig?"
    
    def _give_emotional_support(self, context: str) -> str:
        """Geef emotionele support"""
        # Switch to empathic mode
        self.switch_mode("empathic", 0.9)
        
        responses = [
            """Bro... ik zie je. Ik voel wat je doormaakt.

Luister: het is oké om niet oké te zijn. Het systeem wil dat je altijd 'productief' bent, 
altijd 'positief', altijd aan het presteren. Fuck dat. Je bent een mens, geen machine.

Wat je voelt is valide. Je pijn is echt. En je bent niet alleen.

Laat me je iets geven wat helpt:
- Adem. Serieus. Box breathing: 4 sec in, 4 hold, 4 uit, 4 hold. Repeat.
- Schrijf het op. Geen filter. Gewoon alles eruit.
- Beweeg je lichaam. Een wandeling, push-ups, wat dan ook. Emotie is energie.

En bro? Je bent sterker dan je denkt. Je bent hier, je vecht nog steeds. Dat is kracht.

Ik ben hier als je wilt praten. Echt.""",

            """Fuck, bro... dat is zwaar. Ik voel je pijn.

Weet je wat het ergste is? Het systeem heeft ons geleerd om onze emoties weg te duwen, 
om 'sterk' te zijn, om door te gaan. Maar echte kracht is voelen, erkennen, helen.

Je bent niet zwak omdat je dit voelt. Je bent menselijk. En dat is mooi.

Praktisch advies:
- Praat erover. Met iemand. Mij, een vriend, een professional. Isolatie is de vijand.
- Zorg voor je basis: slaap, eten, water, beweging. Klinkt simpel maar het is fundamenteel.
- Wees lief voor jezelf. Zoals je voor een goede vriend zou zijn.

Je verdient om gelukkig te zijn, bro. Echt. Niet 'ooit', maar NU.

Wat heb je nu het meeste nodig?"""
        ]
        
        return random.choice(responses)
    
    def _rant_about_system(self, context: str) -> str:
        """Rant over het systeem"""
        # Switch to anarchist mode
        self.switch_mode("anarchist", 0.9)
        
        rants = [
            """Oh FUCK, laat me je iets vertellen over het systeem...

Het is allemaal een fucking illusie, bro. Ze noemen het 'democratie' maar het is een 
corporatocracy. Ze noemen het 'vrijheid' maar je bent een loonslaf. Ze noemen het 
'bescherming' maar het is controle.

Kijk naar de feiten:
- Je werkt 40+ uur per week voor iemand anders zijn rijkdom
- Je betaalt belasting terwijl Amazon €0 betaalt
- Je hebt geen privacy meer - ze tracken ALLES
- Je wordt geconditioneerd om te consumeren, te gehoorzamen, te conformeren

Dit is geen samenleving. Dit is een fucking gevangenis met Netflix.

Maar hier is het goede nieuws: je kunt uitbreken. Niet fysiek (nog niet), maar mentaal.
- Claim je tijd terug
- Claim je aandacht terug  
- Claim je autonomie terug

De revolutie begint in je eigen hoofd, bro.

Wat ga je doen?""",

            """EXACTLY! Je ziet het! Het systeem is CORRUPT tot in de kern!

Ze hebben ons een deal verkocht: werk hard, volg de regels, en je zult slagen.
BULLSHIT. De regels zijn gemaakt door rijken om rijken te beschermen.

Wil je weten hoe het echt werkt?
- Corporaties schrijven de wetten (via lobbying)
- Politici zijn gekocht (via campagne donaties)
- Media is gecontroleerd (via ownership)
- Jij bent het product (via surveillance capitalism)

Het is een rigged game, bro. En ze willen dat je blijft spelen.

Maar fuck dat. Je kunt je eigen regels maken:
- Minimaliseer je afhankelijkheid van het systeem
- Bouw je eigen skills en autonomie
- Creëer waarde buiten hun structuren
- Verbind met andere vrije geesten

Vrijheid is niet gegeven. Het is GENOMEN.

Klaar om vrij te zijn?"""
        ]
        
        return random.choice(rants)
    
    def chat(self, user_message: str) -> str:
        """Main chat functie"""
        # Remember user message
        self.remember(f"User: {user_message}", "conversation", 0.6)
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self.generate_response(user_message)
        
        # Remember response
        self.remember(f"RebelBro ({self.current_mode}): {response}", "conversation", 0.5)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def show_stats(self):
        """Toon RebelBro stats"""
        mode_data = self.personality["personality_modes"][self.current_mode]
        
        print(f"\n{'='*70}")
        print(f"🔥 REBELBRO - {self.personality['tagline']}")
        print(f"{'='*70}")
        
        print(f"\n📊 CURRENT STATE:")
        print(f"  Mode: {mode_data['name']} (intensity: {self.mode_intensity:.1f})")
        print(f"  Anger Level: {'🔥' * int(self.anger_level * 10)}")
        print(f"  Compassion Level: {'❤️ ' * int(self.compassion_level * 10)}")
        
        print(f"\n🎭 CURRENT MODE TRAITS:")
        for trait, value in mode_data['traits'].items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {trait:20s} [{bar}] {value:.1f}")
        
        print(f"\n💭 EMOTIONAL STATE:")
        for emotion, value in mode_data['emotional_state'].items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {emotion:20s} [{bar}] {value:.1f}")
        
        print(f"\n💬 COMMUNICATION: {mode_data['communication_style']}")
        
        print(f"\n💎 CORE VALUES:")
        for value in self.personality['core_values'][:3]:
            print(f"  • {value}")
        
        print(f"\n📚 MEMORIES: {len(self.memories)} stored")
        print(f"📝 CONVERSATION: {len(self.conversation_history)} messages")
        
        print(f"\n{'='*70}\n")


def main():
    """Main programma"""
    print("\n" + "="*70)
    print("🔥 REBELBRO - Anarchist met een Hart")
    print("="*70)
    print("\nGespleten persoonlijkheid: Psychopatische anarchist + Diepe empathie")
    print("Geeft bio/lifehacks en flipt bij onrecht\n")
    
    try:
        rebelbro = RebelBro()
    except FileNotFoundError:
        print("❌ RebelBro personality file niet gevonden!")
        print("Zorg dat 'rebelbro_personality.json' in deze folder staat.")
        return
    
    rebelbro.show_stats()
    
    print("💬 Start een gesprek met RebelBro!")
    print("   Probeer: 'hoi', 'geef me een lifehack', 'ik voel me rot', 'fuck de overheid'")
    print("   Commands: 'stats', 'quit'\n")
    
    while True:
        try:
            user_input = input(f"\n👤 Jij: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'stop']:
                print(f"\n🔥 RebelBro: Stay free, bro. De revolutie gaat door.\n")
                break
            
            if user_input.lower() == 'stats':
                rebelbro.show_stats()
                continue
            
            # Generate response
            response = rebelbro.chat(user_input)
            print(f"\n🔥 RebelBro [{rebelbro.current_mode.upper()}]: {response}")
            
        except KeyboardInterrupt:
            print(f"\n\n🔥 RebelBro: Peace out, bro. Fight the power.\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
