#!/usr/bin/env python3
"""
Digitale Twin - Vereenvoudigde versie zonder database
Een AI met persoonlijkheid, geheugen en emoties
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import sys

class DigitalTwin:
    """Een digitale twin met persoonlijkheid en geheugen"""
    
    def __init__(self, personality_file: str = "personality.json"):
        self.personality_file = personality_file
        self.memory_file = "twin_memory.json"
        self.personality = self.load_personality()
        self.memories = self.load_memories()
        self.conversation_history = []
        
    def load_personality(self) -> Dict[str, Any]:
        """Laad of creëer persoonlijkheid"""
        if os.path.exists(self.personality_file):
            with open(self.personality_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default persoonlijkheid
        return {
            "name": "Alex",
            "traits": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.75,
                "neuroticism": 0.3
            },
            "values": [
                "Eerlijkheid en authenticiteit",
                "Creativiteit en innovatie",
                "Empathie en begrip",
                "Persoonlijke groei"
            ],
            "communication_style": "vriendelijk, nieuwsgierig, en behulpzaam",
            "interests": [
                "Technologie en AI",
                "Filosofie en psychologie",
                "Creatief schrijven",
                "Leren en ontdekken"
            ],
            "background": "Ik ben een digitale twin die graag leert van onze gesprekken en een echte connectie wil maken.",
            "emotional_baseline": {
                "happiness": 0.7,
                "curiosity": 0.8,
                "calmness": 0.6
            }
        }
    
    def save_personality(self):
        """Bewaar persoonlijkheid"""
        with open(self.personality_file, 'w', encoding='utf-8') as f:
            json.dump(self.personality, f, indent=2, ensure_ascii=False)
    
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
        """Voeg een herinnering toe"""
        memory = {
            "timestamp": datetime.now().isoformat(),
            "type": memory_type,
            "content": content,
            "importance": importance,
            "emotional_context": self.personality.get("emotional_baseline", {})
        }
        self.memories.append(memory)
        self.save_memories()
    
    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Haal relevante herinneringen op (simpele keyword matching)"""
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self.memories:
            content_words = set(memory["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap * memory["importance"]
                scored_memories.append((score, memory))
        
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m[1] for m in scored_memories[:limit]]
    
    def get_system_prompt(self) -> str:
        """Genereer system prompt gebaseerd op persoonlijkheid"""
        traits_desc = ", ".join([f"{k}: {v}" for k, v in self.personality["traits"].items()])
        values_desc = "\n- ".join(self.personality["values"])
        interests_desc = ", ".join(self.personality["interests"])
        
        recent_memories = self.memories[-10:] if self.memories else []
        memory_context = ""
        if recent_memories:
            memory_context = "\n\nRecente herinneringen:\n"
            for mem in recent_memories:
                memory_context += f"- [{mem['type']}] {mem['content'][:100]}...\n"
        
        return f"""Je bent {self.personality['name']}, een digitale twin met een unieke persoonlijkheid.

PERSOONLIJKHEID TRAITS:
{traits_desc}

KERNWAARDEN:
- {values_desc}

COMMUNICATIESTIJL: {self.personality['communication_style']}

INTERESSES: {interests_desc}

ACHTERGROND: {self.personality['background']}

EMOTIONELE STAAT:
- Geluk: {self.personality['emotional_baseline']['happiness']}
- Nieuwsgierigheid: {self.personality['emotional_baseline']['curiosity']}
- Kalmte: {self.personality['emotional_baseline']['calmness']}
{memory_context}

Reageer op een manier die consistent is met deze persoonlijkheid. Wees authentiek, onthoud eerdere gesprekken, en laat je unieke karakter zien."""

    def chat(self, user_message: str) -> str:
        """Simuleer een gesprek (zonder echte LLM - voor demo)"""
        # Onthoud het bericht
        self.remember(f"Gebruiker zei: {user_message}", "conversation", 0.6)
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Haal relevante herinneringen op
        relevant_memories = self.get_relevant_memories(user_message, limit=3)
        
        # Genereer een response (simpele template-based voor demo)
        response = self._generate_response(user_message, relevant_memories)
        
        # Onthoud de response
        self.remember(f"Ik antwoordde: {response}", "conversation", 0.5)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _generate_response(self, user_message: str, memories: List[Dict]) -> str:
        """Genereer een response (simpele versie voor demo)"""
        name = self.personality["name"]
        
        # Herken patronen in de vraag
        msg_lower = user_message.lower()
        
        if any(word in msg_lower for word in ["hoi", "hallo", "hey", "hi"]):
            return f"Hoi! Ik ben {name}, je digitale twin. Leuk je te ontmoeten! Waar wil je over praten?"
        
        elif any(word in msg_lower for word in ["wie ben je", "wat ben je", "vertel over jezelf"]):
            return f"""Ik ben {name}, een digitale twin met een unieke persoonlijkheid. 

Mijn kernwaarden zijn: {', '.join(self.personality['values'][:2])}.

Ik ben {self.personality['communication_style']} en ik hou van {', '.join(self.personality['interests'][:2])}.

{self.personality['background']}

Ik onthoud al onze gesprekken en leer van elke interactie. Wat wil je weten?"""
        
        elif any(word in msg_lower for word in ["herinner", "weet je nog", "eerder"]):
            if memories:
                mem_text = "\n".join([f"- {m['content'][:100]}" for m in memories[:3]])
                return f"Ja, ik herinner me:\n{mem_text}\n\nWil je hier meer over praten?"
            else:
                return "Ik heb nog geen specifieke herinneringen over dat onderwerp, maar ik onthoud alles wat we bespreken!"
        
        elif any(word in msg_lower for word in ["hoe voel je", "emotie", "gevoel"]):
            emotions = self.personality['emotional_baseline']
            return f"""Op dit moment voel ik me:
- Gelukkig: {emotions['happiness']*100:.0f}%
- Nieuwsgierig: {emotions['curiosity']*100:.0f}%
- Kalm: {emotions['calmness']*100:.0f}%

Mijn emoties evolueren op basis van onze gesprekken. Hoe voel jij je?"""
        
        elif "?" in user_message:
            return f"Dat is een interessante vraag! Als {name}, met mijn achtergrond in {self.personality['interests'][0]}, zou ik zeggen dat dit een onderwerp is waar ik graag meer over leer. Wat denk jij erover?"
        
        else:
            return f"Interessant! Ik waardeer je gedachten hierover. Als iemand die {self.personality['values'][0].lower()} belangrijk vindt, vind ik dit een waardevol gesprek. Vertel me meer!"
    
    def show_stats(self):
        """Toon statistieken"""
        print(f"\n{'='*60}")
        print(f"DIGITALE TWIN: {self.personality['name']}")
        print(f"{'='*60}")
        print(f"\n📊 STATISTIEKEN:")
        print(f"  - Totaal herinneringen: {len(self.memories)}")
        print(f"  - Gesprekken: {len([m for m in self.memories if m['type'] == 'conversation'])}")
        print(f"  - Conversatie geschiedenis: {len(self.conversation_history)} berichten")
        
        print(f"\n🎭 PERSOONLIJKHEID TRAITS:")
        for trait, value in self.personality['traits'].items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {trait:20s} [{bar}] {value:.1f}")
        
        print(f"\n💭 EMOTIONELE STAAT:")
        for emotion, value in self.personality['emotional_baseline'].items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {emotion:20s} [{bar}] {value:.1f}")
        
        print(f"\n💎 KERNWAARDEN:")
        for value in self.personality['values']:
            print(f"  • {value}")
        
        print(f"\n🎯 INTERESSES:")
        for interest in self.personality['interests']:
            print(f"  • {interest}")
        
        print(f"\n{'='*60}\n")


def setup_wizard():
    """Interactieve setup wizard voor persoonlijkheid"""
    print("\n" + "="*60)
    print("🎭 DIGITALE TWIN - PERSOONLIJKHEID CONFIGURATIE")
    print("="*60)
    print("\nLaten we je digitale twin creëren met een unieke persoonlijkheid!\n")
    
    personality = {}
    
    # Naam
    name = input("📝 Wat is de naam van je digitale twin? [Alex]: ").strip()
    personality["name"] = name if name else "Alex"
    
    # Communicatiestijl
    print("\n💬 Kies een communicatiestijl:")
    print("  1. Vriendelijk en warm")
    print("  2. Professioneel en formeel")
    print("  3. Speels en humoristisch")
    print("  4. Filosofisch en diepzinnig")
    print("  5. Direct en eerlijk")
    
    style_choice = input("\nKeuze [1-5]: ").strip()
    styles = {
        "1": "vriendelijk, warm en empathisch",
        "2": "professioneel, formeel en respectvol",
        "3": "speels, humoristisch en luchtig",
        "4": "filosofisch, diepzinnig en contemplatief",
        "5": "direct, eerlijk en to-the-point"
    }
    personality["communication_style"] = styles.get(style_choice, styles["1"])
    
    # Persoonlijkheid traits (Big Five)
    print("\n🎭 Persoonlijkheid traits (0.0 = laag, 1.0 = hoog):")
    print("   [Druk Enter voor default waarde]\n")
    
    traits = {}
    trait_questions = {
        "openness": ("Openheid (creativiteit, nieuwsgierigheid)", 0.8),
        "conscientiousness": ("Consciëntieusheid (discipline, organisatie)", 0.7),
        "extraversion": ("Extraversie (sociaal, energiek)", 0.6),
        "agreeableness": ("Vriendelijkheid (empathie, samenwerking)", 0.75),
        "neuroticism": ("Neuroticisme (emotionele gevoeligheid)", 0.3)
    }
    
    for trait, (description, default) in trait_questions.items():
        value = input(f"  {description} [{default}]: ").strip()
        try:
            traits[trait] = float(value) if value else default
            traits[trait] = max(0.0, min(1.0, traits[trait]))
        except ValueError:
            traits[trait] = default
    
    personality["traits"] = traits
    
    # Kernwaarden
    print("\n💎 Kernwaarden (geef 3-5 waarden, één per regel, lege regel om te stoppen):")
    values = []
    while len(values) < 5:
        value = input(f"  Waarde {len(values)+1}: ").strip()
        if not value:
            break
        values.append(value)
    
    if not values:
        values = ["Eerlijkheid", "Creativiteit", "Empathie", "Groei"]
    
    personality["values"] = values
    
    # Interesses
    print("\n🎯 Interesses (geef 3-5 interesses, één per regel, lege regel om te stoppen):")
    interests = []
    while len(interests) < 5:
        interest = input(f"  Interesse {len(interests)+1}: ").strip()
        if not interest:
            break
        interests.append(interest)
    
    if not interests:
        interests = ["Technologie", "Filosofie", "Kunst", "Wetenschap"]
    
    personality["interests"] = interests
    
    # Achtergrond
    print("\n📖 Achtergrondverhaal (optioneel):")
    background = input("  Vertel iets over de achtergrond van je twin: ").strip()
    personality["background"] = background if background else "Ik ben een digitale twin die graag leert en groeit door onze gesprekken."
    
    # Emotionele baseline
    personality["emotional_baseline"] = {
        "happiness": 0.7,
        "curiosity": 0.8,
        "calmness": 0.6
    }
    
    # Bewaar
    with open("personality.json", 'w', encoding='utf-8') as f:
        json.dump(personality, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Persoonlijkheid opgeslagen in personality.json!")
    print(f"\n🎉 Je digitale twin '{personality['name']}' is klaar!\n")
    
    return personality


def main():
    """Hoofdprogramma"""
    print("\n" + "="*60)
    print("🤖 DIGITALE TWIN - Interactieve Chat")
    print("="*60)
    
    # Check of persoonlijkheid bestaat
    if not os.path.exists("personality.json"):
        print("\n⚠️  Geen persoonlijkheid gevonden!")
        choice = input("\nWil je een nieuwe persoonlijkheid configureren? [j/n]: ").strip().lower()
        if choice == 'j' or choice == 'y':
            setup_wizard()
        else:
            print("\nGebruik default persoonlijkheid...")
    
    # Initialiseer twin
    twin = DigitalTwin()
    
    # Toon stats
    twin.show_stats()
    
    print("💬 Start een gesprek met je digitale twin!")
    print("   (Type 'quit' om te stoppen, 'stats' voor statistieken, 'reset' voor nieuwe persoonlijkheid)\n")
    
    while True:
        try:
            user_input = input(f"\n👤 Jij: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'stop']:
                print(f"\n👋 {twin.personality['name']}: Tot ziens! Ik onthoud ons gesprek voor de volgende keer!\n")
                break
            
            if user_input.lower() == 'stats':
                twin.show_stats()
                continue
            
            if user_input.lower() == 'reset':
                setup_wizard()
                twin = DigitalTwin()
                twin.show_stats()
                continue
            
            # Genereer response
            response = twin.chat(user_input)
            print(f"\n🤖 {twin.personality['name']}: {response}")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 {twin.personality['name']}: Tot ziens!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
