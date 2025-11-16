using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Agente ML avançado que intercepta as chamadas de observação e as processa no Rust
/// </summary>
public class AdvancedRustMLAgent : Agent
{
    private CognitionLearnBridge m_CognitionBridge;

    void Start()
    {
        // Garante que o bridge existe
        if (m_CognitionBridge == null)
        {
            m_CognitionBridge = FindObjectOfType<CognitionLearnBridge>();
            if (m_CognitionBridge == null)
            {
                GameObject bridgeObj = new GameObject("CognitionLearnBridge");
                m_CognitionBridge = bridgeObj.AddComponent<CognitionLearnBridge>();
            }
        }

        // Associa este agente ao bridge
        if (m_CognitionBridge != null)
        {
            m_CognitionBridge.SetAgent(this);
        }
    }

    /// <summary>
    /// Override do método CollectObservations para interceptar e enviar para o Rust
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Primeiro adicionamos as observações normais como faríamos normalmente
        // Exemplo - substitua com suas próprias observações:
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.rotation);

        // Em vez de enviar diretamente para o servidor Python,
        // nós interceptamos e enviamos para o Rust
        // Neste exemplo, vamos capturar as observações e enviar para o Rust

        // Para interceptar efetivamente, precisamos capturar as observações antes de adicioná-las
        // Esta é uma implementação simplificada que demonstra o conceito
        float[] exampleObservations = new float[] {
            transform.position.x,
            transform.position.y,
            transform.position.z,
            transform.rotation.eulerAngles.x,
            transform.rotation.eulerAngles.y,
            transform.rotation.eulerAngles.z
        };

        // Processa as observações no Rust
        if (m_CognitionBridge != null)
        {
            string result = m_CognitionBridge.ProcessObservations(exampleObservations);
            Debug.Log("Resultado do Rust para observações: " + result);
        }
        else
        {
            Debug.LogError("CognitionLearnBridge não encontrado!");
        }
    }

    /// <summary>
    /// Override do método OnActionReceived para demonstrar o retorno das ações do Rust (parte 2)
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Recebe as ações do Rust (parte 2) - para validar que está funcionando, vamos testar com valores -9
        if (m_CognitionBridge != null)
        {
            // Para a parte 2, obtemos ações de teste com valores -9 para validação
            float[] testActions = m_CognitionBridge.GetTestActions(3); // 3 ações de teste

            if (testActions.Length > 0 && testActions[0] == -9.0f)
            {
                Debug.Log("Validação da parte 2 bem-sucedida: Ações recebidas do Rust contêm valor -9: " +
                         string.Join(", ", testActions));
            }
        }

        // Aplica as ações normais
        base.OnActionReceived(actions);

        // Exemplo de como as ações vêm do parâmetro 'actions'
        Debug.Log($"Ações recebidas normalmente: contínuas=[{string.Join(", ", actions.ContinuousActions)}], discretas=[{string.Join(", ", actions.DiscreteActions)}]");
    }
}