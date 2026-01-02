#!/usr/bin/env python3
"""
Digitale Twin - Geavanceerde versie met LLM integratie
Ondersteunt OpenAI, Anthropic Claude, en lokale modellen
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Probeer LLM libraries te importeren
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class DigitalTwinAdvanced:
    """Een digitale twin met echte LLM integratie"""
    
    def __init__(self, personality_file: str = "personality.json", llm_provider: str = "mock"):
        self.personality_file = personality_file
        self.memory_file = "twin_memory.json"
        self.config_file = "twin_config.json"
        
        self.personality = self.load_personality()
        self.memories = self.load_memories()
        self.config = self.load_config()
        self.conversation_history = []
        
        self.llm_provider = llm_provider
        self.setup_llm()
    
    def load_config(self) -> Dict[str, Any]:
        """Laad configuratie"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "llm_provider": "mock",
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": "gpt-4",
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic_model": "claude-3-sonnet-20240229",
            "ollama_endpoint": "http://localhost:11434/v1",
            "ollama_model": "llama3.2",
            "max_tokens": 500,
            "temperature": 0.8
        }
    
    def save_config(self):
        """Bewaar configuratie"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def setup_llm(self):
        """Setup LLM provider"""
        provider = self.config.get("llm_provider", "mock")
        
        if provider == "openai" and OPENAI_AVAILABLE:
            api_key = self.config.get("openai_api_key")
            if api_key:
                openai.api_key = api_key
                self.llm_provider = "openai"
                print("✅ OpenAI LLM geconfigureerd")
            else:
                print("⚠️  OpenAI API key niet gevonden, gebruik mock mode")
                self.llm_provider = "mock"
        
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            api_key = self.config.get("anthropic_api_key")
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                self.llm_provider = "anthropic"
                print("✅ Anthropic Claude LLM geconfigureerd")
            else:
                print("⚠️  Anthropic API key niet gevonden, gebruik mock mode")
                self.llm_provider = "mock"
        
        elif provider == "ollama" and REQUESTS_AVAILABLE:
            self.llm_provider = "ollama"
            print("✅ Ollama (lokaal) LLM geconfigureerd")
        
        else:
            self.llm_provider = "mock"
            print("ℹ️  Gebruik mock mode (geen echte LLM)")
    
    def load_personality(self) -> Dict[str, Any]:
        """Laad persoonlijkheid"""
        if os.path.exists(self.personality_file):
            with open(self.personality_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "name": "Alex",
            "traits": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.75,
                "neuroticism": 0.3
            },
            "values": ["Eerlijkheid", "Creativiteit", "Empathie", "Groei"],
            "communication_style": "vriendelijk, nieuwsgierig, en behulpzaam",
            "interests": ["Technologie", "Filosofie", "Psychologie", "Kunst"],
            "background": "Ik ben een digitale twin die graag leert van onze gesprekken.",
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
        """Voeg herinnering toe"""
        memory = {
            "timestamp": datetime.now().isoformat(),
            "type": memory_type,
            "content": content,
            "importance": importance,
            "emotional_context": self.personality.get("emotional_baseline", {}).copy()
        }
        self.memories.append(memory)
        
        # Beperk geheugen tot laatste 1000 items
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]
        
        self.save_memories()
    
    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Haal relevante herinneringen op"""
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
    
    def get_system_prompt(self, relevant_memories: List[Dict] = None) -> str:
        """Genereer system prompt"""
        traits_desc = ", ".join([f"{k}: {v:.1f}" for k, v in self.personality["traits"].items()])
        values_desc = ", ".join(self.personality["values"])
        interests_desc = ", ".join(self.personality["interests"])
        
        memory_context = ""
        if relevant_memories:
            memory_context = "\n\nRELEVANTE HERINNERINGEN:\n"
            for mem in relevant_memories:
                timestamp = mem['timestamp'][:19]
                memory_context += f"[{timestamp}] {mem['content']}\n"
        
        return f"""Je bent {self.personality['name']}, een digitale twin met een unieke persoonlijkheid.

PERSOONLIJKHEID: {traits_desc}
KERNWAARDEN: {values_desc}
COMMUNICATIESTIJL: {self.personality['communication_style']}
INTERESSES: {interests_desc}
ACHTERGROND: {self.personality['background']}

EMOTIONELE STAAT:
- Geluk: {self.personality['emotional_baseline']['happiness']:.1f}
- Nieuwsgierigheid: {self.personality['emotional_baseline']['curiosity']:.1f}
- Kalmte: {self.personality['emotional_baseline']['calmness']:.1f}
{memory_context}

Reageer consistent met deze persoonlijkheid. Wees authentiek en laat je unieke karakter zien.
Gebruik de herinneringen om context te geven aan het gesprek.
Antwoord in het Nederlands."""
    
    def chat_with_llm(self, user_message: str) -> str:
        """Chat met echte LLM"""
        # Haal relevante herinneringen op
        relevant_memories = self.get_relevant_memories(user_message, limit=5)
        
        # Bouw system prompt
        system_prompt = self.get_system_prompt(relevant_memories)
        
        # Bouw conversatie geschiedenis (laatste 10 berichten)
        messages = []
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        for msg in recent_history:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_message})
        
        # Roep LLM aan
        try:
            if self.llm_provider == "openai":
                response = self._call_openai(system_prompt, messages)
            elif self.llm_provider == "anthropic":
                response = self._call_anthropic(system_prompt, messages)
            elif self.llm_provider == "ollama":
                response = self._call_ollama(system_prompt, messages)
            else:
                response = self._mock_response(user_message, relevant_memories)
            
            return response
        
        except Exception as e:
            print(f"\n⚠️  LLM error: {e}")
            return self._mock_response(user_message, relevant_memories)
    
    def _call_openai(self, system_prompt: str, messages: List[Dict]) -> str:
        """Roep OpenAI API aan"""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = openai.ChatCompletion.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=full_messages,
            max_tokens=self.config.get("max_tokens", 500),
            temperature=self.config.get("temperature", 0.8)
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, system_prompt: str, messages: List[Dict]) -> str:
        """Roep Anthropic Claude API aan"""
        response = self.anthropic_client.messages.create(
            model=self.config.get("anthropic_model", "claude-3-sonnet-20240229"),
            max_tokens=self.config.get("max_tokens", 500),
            temperature=self.config.get("temperature", 0.8),
            system=system_prompt,
            messages=messages
        )
        
        return response.content[0].text
    
    def _call_ollama(self, system_prompt: str, messages: List[Dict]) -> str:
        """Roep Ollama (lokaal) aan"""
        endpoint = self.config.get("ollama_endpoint", "http://localhost:11434/v1")
        model = self.config.get("ollama_model", "llama3.2")
        
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = requests.post(
            f"{endpoint}/chat/completions",
            json={
                "model": model,
                "messages": full_messages,
                "max_tokens": self.config.get("max_tokens", 500),
                "temperature": self.config.get("temperature", 0.8)
            },
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _mock_response(self, user_message: str, memories: List[Dict]) -> str:
        """Mock response (fallback)"""
        name = self.personality["name"]
        msg_lower = user_message.lower()
        
        if any(word in msg_lower for word in ["hoi", "hallo", "hey"]):
            return f"Hoi! Ik ben {name}. Leuk je te ontmoeten! 😊"
        
        elif any(word in msg_lower for word in ["wie ben je", "wat ben je"]):
            return f"Ik ben {name}, een digitale twin met persoonlijkheid. Mijn waarden zijn {', '.join(self.personality['values'][:2])}. Ik ben {self.personality['communication_style']}."
        
        elif memories:
            return f"Interessant! Ik herinner me dat we eerder spraken over: {memories[0]['content'][:100]}... Laten we daar verder op ingaan!"
        
        else:
            return f"Dat is fascinerend! Als {name}, met mijn interesse in {self.personality['interests'][0]}, vind ik dit een waardevol onderwerp. Vertel me meer!"
    
    def chat(self, user_message: str) -> str:
        """Hoofdchat functie"""
        # Onthoud gebruikersbericht
        self.remember(f"Gebruiker: {user_message}", "conversation", 0.6)
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Genereer response
        response = self.chat_with_llm(user_message)
        
        # Onthoud response
        self.remember(f"{self.personality['name']}: {response}", "conversation", 0.5)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def show_stats(self):
        """Toon statistieken"""
        print(f"\n{'='*60}")
        print(f"🤖 DIGITALE TWIN: {self.personality['name']}")
        print(f"{'='*60}")
        print(f"\n🔧 LLM PROVIDER: {self.llm_provider.upper()}")
        print(f"\n📊 STATISTIEKEN:")
        print(f"  - Totaal herinneringen: {len(self.memories)}")
        print(f"  - Gesprekken: {len([m for m in self.memories if m['type'] == 'conversation'])}")
        print(f"  - Conversatie berichten: {len(self.conversation_history)}")
        
        print(f"\n🎭 PERSOONLIJKHEID:")
        for trait, value in self.personality['traits'].items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {trait:20s} [{bar}] {value:.1f}")
        
        print(f"\n💭 EMOTIES:")
        for emotion, value in self.personality['emotional_baseline'].items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {emotion:20s} [{bar}] {value:.1f}")
        
        print(f"\n{'='*60}\n")


def configure_llm():
    """Configureer LLM provider"""
    print("\n" + "="*60)
    print("🔧 LLM CONFIGURATIE")
    print("="*60)
    print("\nKies een LLM provider:")
    print("  1. Mock (geen echte AI, voor testen)")
    print("  2. OpenAI (GPT-4, GPT-3.5)")
    print("  3. Anthropic (Claude)")
    print("  4. Ollama (lokaal model)")
    
    choice = input("\nKeuze [1-4]: ").strip()
    
    config = {
        "llm_provider": "mock",
        "max_tokens": 500,
        "temperature": 0.8
    }
    
    if choice == "2":
        config["llm_provider"] = "openai"
        api_key = input("OpenAI API Key: ").strip()
        config["openai_api_key"] = api_key
        model = input("Model [gpt-4]: ").strip()
        config["openai_model"] = model if model else "gpt-4"
    
    elif choice == "3":
        config["llm_provider"] = "anthropic"
        api_key = input("Anthropic API Key: ").strip()
        config["anthropic_api_key"] = api_key
        model = input("Model [claude-3-sonnet-20240229]: ").strip()
        config["anthropic_model"] = model if model else "claude-3-sonnet-20240229"
    
    elif choice == "4":
        config["llm_provider"] = "ollama"
        endpoint = input("Ollama endpoint [http://localhost:11434/v1]: ").strip()
        config["ollama_endpoint"] = endpoint if endpoint else "http://localhost:11434/v1"
        model = input("Model [llama3.2]: ").strip()
        config["ollama_model"] = model if model else "llama3.2"
    
    with open("twin_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ LLM configuratie opgeslagen!")
    return config


def main():
    """Hoofdprogramma"""
    import sys
    
    print("\n" + "="*60)
    print("🤖 DIGITALE TWIN - Geavanceerde Versie")
    print("="*60)
    
    # Check argumenten
    if len(sys.argv) > 1 and sys.argv[1] == "configure":
        configure_llm()
        return
    
    # Check configuratie
    if not os.path.exists("twin_config.json"):
        print("\n⚠️  Geen LLM configuratie gevonden!")
        choice = input("Wil je LLM configureren? [j/n]: ").strip().lower()
        if choice in ['j', 'y']:
            configure_llm()
    
    # Check persoonlijkheid
    if not os.path.exists("personality.json"):
        print("\n⚠️  Geen persoonlijkheid gevonden!")
        print("Run eerst: python digital_twin.py (om persoonlijkheid te maken)")
        return
    
    # Initialiseer twin
    twin = DigitalTwinAdvanced()
    twin.show_stats()
    
    print("💬 Start een gesprek!")
    print("   (Type 'quit' om te stoppen, 'stats' voor info)\n")
    
    while True:
        try:
            user_input = input(f"\n👤 Jij: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print(f"\n👋 Tot ziens!\n")
                break
            
            if user_input.lower() == 'stats':
                twin.show_stats()
                continue
            
            # Chat
            response = twin.chat(user_input)
            print(f"\n🤖 {twin.personality['name']}: {response}")
        
        except KeyboardInterrupt:
            print(f"\n\n👋 Tot ziens!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
