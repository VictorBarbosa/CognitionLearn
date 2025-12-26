using System.Collections;
using UnityEngine;

public class SizeVerificationTest : MonoBehaviour
{
    private CognitionLearnBridge bridge;

    void Start()
    {
        // Cria o bridge para testar a integração
        if (bridge == null)
        {
            bridge = FindObjectOfType<CognitionLearnBridge>();
            if (bridge == null)
            {
                GameObject bridgeObj = new GameObject("CognitionLearnBridge");
                bridge = bridgeObj.AddComponent<CognitionLearnBridge>();
            }
        }

        // Executa o teste de verificação de tamanho
        StartCoroutine(RunSizeVerificationTests());
    }

    private IEnumerator RunSizeVerificationTests()
    {
        yield return new WaitForSeconds(1); // Espera um frame para garantir que tudo esteja inicializado

        // Teste 1: Array com 5 elementos
        Debug.Log("=== Teste 1: Array com 5 elementos ===");
        float[] testArray1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        yield return TestArraySize(testArray1, 5);

        // Teste 2: Array com 10 elementos
        Debug.Log("\n=== Teste 2: Array com 10 elementos ===");
        float[] testArray2 = new float[10];
        for (int i = 0; i < 10; i++)
        {
            testArray2[i] = i * 1.5f;
        }
        yield return TestArraySize(testArray2, 10);

        // Teste 3: Array com 1 elemento
        Debug.Log("\n=== Teste 3: Array com 1 elemento ===");
        float[] testArray3 = new float[] { 42.0f };
        yield return TestArraySize(testArray3, 1);

        // Teste 4: Array vazio (caso limite)
        Debug.Log("\n=== Teste 4: Array vazio ===");
        float[] testArray4 = new float[0];
        yield return TestArraySize(testArray4, 0);

        Debug.Log("\n=== Testes de verificação de tamanho concluídos ===");
    }

    private IEnumerator TestArraySize(float[] testArray, int expectedSize)
    {
        yield return new WaitForSeconds(0.5f); // Pequeno intervalo entre testes

        Debug.Log($"Enviando array com {testArray.Length} elementos para o Rust...");
        
        string result = bridge.ProcessObservations(testArray);
        
        Debug.Log($"Resultado do Rust: {result}");
        
        // Verifica se o tamanho no resultado do Rust corresponde ao tamanho do array enviado
        string expectedSubstring = $"tamanho das observacoes e: {expectedSize}";
        
        if (result.Contains(expectedSubstring))
        {
            Debug.Log($"✅ VERIFICAÇÃO DE TAMANHO: SUCESSO! Tamanho esperado: {expectedSize}, Recebido: {expectedSize}");
        }
        else if (testArray.Length == 0 && result.Contains("Erro: Array de observações nulo ou vazio"))
        {
            Debug.Log("✅ VERIFICAÇÃO DE TAMANHO: SUCESSO! Caso de array vazio tratado corretamente.");
        }
        else
        {
            Debug.Log($"❌ VERIFICAÇÃO DE TAMANHO: FALHOU! Resultado esperado contendo '{expectedSubstring}', mas recebeu: '{result}'");
        }
    }
}