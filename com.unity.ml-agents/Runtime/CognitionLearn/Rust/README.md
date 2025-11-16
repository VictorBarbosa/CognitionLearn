# CognitionLearn

CognitionLearn é um projeto Rust que fornece integração com o ML-Agents do Unity por meio de FFI (Foreign Function Interface), substituindo a comunicação tradicional GRPC com Python.

## Funcionalidades Implementadas

### Parte 1: Interceptação e Envio para Rust

1. **Funções Rust para Processamento de Observações**
   - `process_mlagent_observations(observations: *const f32, length: usize) -> *mut c_char`
   - Recebe observações do ML-Agent como array de floats
   - Retorna string de confirmação com o formato: "Observacoes recebidas, o tamanho das observacoes e: X"

2. **Gerenciamento de Memória**
   - `free_string(ptr: *mut c_char)` - libera memória alocada para strings
   - `free_float_array(data: *mut f32, len: usize)` - libera memória alocada para arrays de floats

3. **Suporte para Parte 2 (envio de ações do Rust para C#)**
   - `get_test_actions(num_actions: usize) -> FloatArray` - retorna array com valores -9 para validação
   - `get_float_array_data(array: FloatArray, index: usize) -> f32` - função auxiliar para debugging

### Implementação C# 

O projeto Unity inclui os seguintes componentes:

1. **CognitionLearnBridge.cs** - Wrapper FFI para chamar funções Rust
2. **AdvancedRustMLAgent.cs** - Agente ML que intercepta `CollectObservations` e envia observações para o Rust
3. **TestRustIntegration.cs** - Testes para validação da integração
4. **SizeVerificationTest.cs** - Testes para verificação de tamanhos

## Como Compilar

```bash
cd CognitionLearn
cargo build --release
```

A biblioteca dinâmica será criada em `target/release/libcognition_learn.dylib` (macOS) / `target/release/libcognition_learn.so` (Linux) / `target/release/cognition_learn.dll` (Windows).

## Como Usar no Unity

1. Copie a biblioteca compilada para a pasta `RustLib` no projeto Unity
2. Adicione o `AdvancedRustMLAgent.cs` a um GameObject no Unity
3. As observações do agente serão automaticamente enviadas para o Rust via FFI
4. O resultado da processamento do Rust será exibido no console do Unity

## Resultado

- [x] Observações são corretamente enviadas do Unity para o Rust
- [x] Confirmação com tamanho das observações é retornada corretamente
- [x] Parte 2 está pronta - funções para envio de ações do Rust para C# implementadas
- [x] Tamanhos de arrays são verificados e correspondem entre C# e Rust
- [x] Validação com valores -9 para a parte 2 está funcionando

## Próximos Passos (Parte 3)

Implementar algoritmo PPO em Rust usando a biblioteca Burn-rs para processar as observações e retornar ações para o Unity.