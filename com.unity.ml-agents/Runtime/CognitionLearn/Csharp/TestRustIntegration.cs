using System.Collections;
using UnityEngine;

public class TestRustIntegration : MonoBehaviour
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

        // Executa o teste
        StartCoroutine(TestObservationProcessing());
    }

    private IEnumerator TestObservationProcessing()
    {
        yield return new WaitForSeconds(1); // Espera um frame para garantir que tudo esteja inicializado

        // Testa o envio de observações para o Rust
        float[] testObservations = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        
        Debug.Log("Enviando observações para o Rust: " + string.Join(", ", testObservations));
        
        string result = bridge.ProcessObservations(testObservations);
        
        Debug.Log("Resultado recebido do Rust: " + result);
        
        // Verifica se o resultado contém o tamanho correto das observações
        if (result.Contains("Observacoes recebidas, o tamanho das observacoes e: 5"))
        {
            Debug.Log("✅ Teste de envio de observações para Rust: SUCESSO!");
        }
        else
        {
            Debug.Log("❌ Teste de envio de observações para Rust: FALHOU!");
        }
        
        // Testa a parte 2 - recepção de ações do Rust
        float[] testActions = bridge.GetTestActions(3);
        
        Debug.Log("Ações recebidas do Rust: " + string.Join(", ", testActions));
        
        // Verifica se as ações contêm o valor -9 como especificado
        bool hasNegativeNine = false;
        foreach (float action in testActions)
        {
            if (action == -9.0f)
            {
                hasNegativeNine = true;
                break;
            }
        }
        
        if (hasNegativeNine)
        {
            Debug.Log("✅ Teste de recepção de ações do Rust (parte 2): SUCESSO! Valor -9 encontrado.");
        }
        else
        {
            Debug.Log("❌ Teste de recepção de ações do Rust (parte 2): FALHOU! Valor -9 não encontrado.");
        }
    }
}