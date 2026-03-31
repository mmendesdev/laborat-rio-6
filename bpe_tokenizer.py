from collections import defaultdict
import re

vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

def get_stats(vocab_dict):
    pairs = defaultdict(int)
    
    for word, freq in vocab_dict.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq
    
    return dict(pairs)

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    
    return v_out

def train_bpe_tokenizer(vocab_dict, num_iterations=5):
    current_vocab = vocab_dict.copy()
    
    print("=== INÍCIO DO TREINAMENTO BPE ===\n")
    print(f"Vocabulário inicial: {current_vocab}\n")
    
    for i in range(num_iterations):
        print(f"--- Iteração {i + 1} ---")
        
        pairs = get_stats(current_vocab)
        
        if not pairs:
            print("Nenhum par encontrado para fusão.\n")
            break
        
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]
        
        print(f"Par mais frequente: {best_pair} (frequência: {best_freq})")
        
        current_vocab = merge_vocab(best_pair, current_vocab)
        
        print(f"Vocabulário após fusão: {current_vocab}\n")
    
    print("=== FIM DO TREINAMENTO BPE ===")
    return current_vocab

def test_wordpiece_tokenizer():
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        
        test_sentence = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
        
        print("\n=== TESTE WORDPIECE ===")
        print(f"Frase original: {test_sentence}")
        
        tokens = tokenizer.tokenize(test_sentence)
        print(f"Tokens WordPiece: {tokens}")
        
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"IDs dos tokens: {token_ids}")
        
        decoded = tokenizer.decode(token_ids)
        print(f"Decodificado: {decoded}")
        
        return tokens
        
    except ImportError:
        print("\nAVISO: Biblioteca transformers não encontrada.")
        print("Instale com: pip install transformers")
        return None
    except Exception as e:
        print(f"\nERRO ao testar WordPiece: {e}")
        return None

def main():
    print("LABORATÓRIO 6 - P2: TOKENIZADOR BPE E WORDPIECE")
    print("=" * 60)
    
    print("\n=== TAREFA 1: MOTOR DE FREQUÊNCIAS ===")
    pairs = get_stats(vocab)
    print(f"Estatísticas dos pares: {pairs}")
    
    es_freq = pairs.get(('e', 's'), 0)
    print(f"Frequência do par ('e', 's'): {es_freq}")
    print(f"Validação: {'CORRETO' if es_freq == 9 else 'INCORRETO'} (esperado: 9)")
    
    print("\n=== TAREFA 2: LOOP DE FUSÃO ===")
    final_vocab = train_bpe_tokenizer(vocab, num_iterations=5)
    
    print("\n=== TAREFA 3: INTEGRAÇÃO WORDPIECE ===")
    test_wordpiece_tokenizer()

if __name__ == "__main__":
    main()
