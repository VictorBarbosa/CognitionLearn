using System;
using System.Runtime.InteropServices;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Agente ML que intercepta as chamadas de observação e as processa no Rust
/// </summary>
public class RustMLAgent : Agent
{
    private CognitionLearnBridge m_CognitionBridge;

    /// <summary>
    /// Override do método CollectObservations para interceptar e enviar para o Rust
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Primeiro coleta as observações normalmente
        base.CollectObservations(sensor);

        // Extrai as observações do sensor para enviar ao Rust
        // Neste exemplo, usamos um método alternativo para obter as observações
        // já que o VectorSensor não expõe diretamente os dados brutos
        
        // Para esta implementação, faremos um interceptação mais direta
        // Adicionamos observações de exemplo para demonstrar o conceito
        float[] exampleObservations = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }; // substituir com observações reais
        
        // Processa as observações no Rust
        if (m_CognitionBridge == null)
        {
            m_CognitionBridge = FindObjectOfType<CognitionLearnBridge>();
            if (m_CognitionBridge == null)
            {
                GameObject bridgeObj = new GameObject("CognitionLearnBridge");
                m_CognitionBridge = bridgeObj.AddComponent<CognitionLearnBridge>();
            }
        }

        if (m_CognitionBridge != null)
        {
            string result = m_CognitionBridge.ProcessObservations(exampleObservations);
            Debug.Log("Resultado do Rust: " + result);
        }
    }

    /// <summary>
    /// Override do método OnActionReceived para demonstrar o retorno das ações do Rust (parte 2)
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Recebe as ações do Rust (parte 2)
        // Por enquanto, vamos implementar o básico
        base.OnActionReceived(actions);

        // Neste ponto, as ações poderiam vir do Rust, mas inicialmente usaremos o comportamento padrão
        // Depois implementaremos a parte 2 onde o Rust envia as ações de volta para cá
    }

    /// <summary>
    /// Método para receber ações do Rust (parte 2)
    /// </summary>
    public void ReceiveActionsFromRust(float[] actions)
    {
        // Este método será chamado pelo Rust para passar as ações
        // Precisamos implementar a lógica de como passar essas ações para o agente
        Debug.Log("Ações recebidas do Rust: " + string.Join(", ", actions));
        
        // Neste exemplo, simplesmente registramos que recebemos as ações
        // Na implementação completa, estas seriam as ações que o agente usaria
    }
}