# Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece

Este projeto implementa o algoritmo Byte Pair Encoding (BPE) do zero e demonstra o uso do tokenizador WordPiece do Hugging Face para processamento de linguagem natural.

## Estrutura do Projeto

- `bpe_tokenizer.py` - Implementação principal do tokenizador BPE e testes com WordPiece
- `requirements.txt` - Dependências Python necessárias

## Tarefas Implementadas

### Tarefa 1: Motor de Frequências

Implementada a função `get_stats()` que conta a frequência de todos os pares adjacentes de caracteres/símbolos no vocabulário. A validação confirma que o par ('e', 's') retorna frequência 9, conforme esperado.

### Tarefa 2: Loop de Fusão

Implementada a função `merge_vocab()` que realiza a fusão do par mais frequente no vocabulário, criando um novo token. O loop principal executa 5 iterações, demonstrando a formação progressiva de tokens morfológicos como o sufixo `est</w>`.

### Tarefa 3: Integração Industrial e WordPiece

Utilização da biblioteca `transformers` do Hugging Face com o tokenizador BERT multilíngue (`bert-base-multilingual-cased`) para demonstrar o funcionamento do WordPiece na prática.

## Sinais de Cerquilha (##) no WordPiece

Os sinais de cerquilha (`##`) nos tokens resultantes do WordPiece indicam que o token é uma **sub-palavra que continua uma palavra anterior**. Esta convenção é fundamental para o funcionamento adequado dos tokenizadores modernos:

- **Tokens sem ##**: Representam o início de uma palavra ou uma palavra completa (ex: "Os", "transform")
- **Tokens com ##**: Representam continuação de uma palavra (ex: "##er", "##mente", "##f")

### Vantagens do Uso de Sub-palavras

1. **Prevenção de Problemas com Vocabulário Desconhecido**: Palavras raras ou nunca vistas podem ser decompostas em sub-palavras conhecidas, evitando o problema de "unknown tokens" ([UNK]) que limitava modelos anteriores.

2. **Eficiência Morfológica**: O modelo pode reconhecer padrões morfológicos como:
   - Sufixos: `##mente` (advérbios), `##ção` (substantivos)
   - Prefixos: `in##`, `des##`
   - Radicais: `transform`, `constit`

3. **Representação Compacta**: Em vez de manter um vocabulário massivo com todas as possíveis palavras, o modelo mantém um vocabulário otimizado de sub-palavras que podem combinar-se para formar qualquer palavra.

4. **Generalização Melhorada**: O modelo aprende relações entre palavras semanticamente relacionadas que compartilham sub-palavras (ex: "transformar", "transformação", "transformador").

## Execução do Projeto

### Pré-requisitos

- Python 3.7+
- Ambiente virtual configurado

### Instalação

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Execução

```bash
python bpe_tokenizer.py
```

## Resultados Esperados

### BPE Implementation

- Validação correta: par ('e', 's') com frequência 9
- Formação de tokens como `est</w>` após 5 iterações
- Demonstração do processo iterativo de fusão de pares

### WordPiece Tokenization

Frase de teste: "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

Tokens resultantes demonstram:

```python
['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform', '##er', 'são', 'in', '##cons', '##tit', '##uc', '##ional', '##mente', 'di', '##f', '##í', '##cei', '##s', 'de', 'aj', '##usta', '##r', '.']
```

## Citação de Uso de IA

Durante o desenvolvimento deste projeto, foi utilizada inteligência artificial generativa (Cascade) para auxiliar na implementação das funções de substituição de string, particularmente:

- **Função `merge_vocab()` (linha 23-31)**: A expressão regular `r'(?<!\S)' + bigram + r'(?!\S)'` para identificação e substituição de bigramas foi gerada com auxílio de IA e posteriormente revisada e validada manualmente. Esta expressão garante que apenas bigramas isolados sejam substituídos, evitando substituições indevidas dentro de outras palavras.
- **Estrutura geral do código**: Organização das funções e documentação inicial foram assistidas por IA.
- **Otimização do código**: Remoção de comentários desnecessários foi sugerida por IA para tornar o código mais conciso.

Todos os trechos gerados por IA foram completamente revisados, testados e validados quanto à correção funcional e aderência aos requisitos do laboratório. A lógica principal do algoritmo BPE e as validações foram implementadas e compreendidas integralmente pelo autor.

## Versionamento

Este projeto utiliza a tag `v1.0` para indicar a versão final a ser avaliada, conforme especificado nas instruções de entrega.

## Referências

- Vaswani, A. et al. (2017). "Attention Is All You Need"
- Devlin, J. et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Hugging Face Transformers Documentation
