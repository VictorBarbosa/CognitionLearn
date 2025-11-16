using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

// Adicionando o namespace para o atributo MonoPInvokeCallback
using AOT;

public class CognitionLearnBridge : MonoBehaviour
{
    // Importa as funções do Rust do novo projeto CognitionLearn
    [DllImport("cognition_learn")]
    private static extern IntPtr process_mlagent_observations(IntPtr observations, int length);

    [DllImport("cognition_learn")]
    private static extern void free_string(IntPtr ptr);

    // Adiciona importações para a parte 2 (envio de ações do Rust para C#)
    [DllImport("cognition_learn")]
    private static extern FloatArray get_test_actions(int num_actions);

    // Define a estrutura correspondente ao FloatArray do Rust
    [StructLayout(LayoutKind.Sequential)]
    private struct FloatArray
    {
        public IntPtr data;
        public int len;
    }

    // Variáveis para controle
    private string lastConfirmation = "";
    private bool isProcessing = false;

    // Fila para armazenar resultados
    private System.Collections.Concurrent.ConcurrentQueue<string> confirmationResults =
        new System.Collections.Concurrent.ConcurrentQueue<string>();

    // Referência ao agente para poder enviar ações de volta
    private Agent m_Agent;

    void Start()
    {
        Debug.Log("CognitionLearnBridge inicializado");
    }

    /// <summary>
    /// Associa o bridge a um agente específico
    /// </summary>
    public void SetAgent(Agent agent)
    {
        m_Agent = agent;
    }

    /// <summary>
    /// Processa as observações do ML-Agent no Rust e retorna uma string de confirmação
    /// </summary>
    public string ProcessObservations(float[] observations)
    {
        if (observations == null || observations.Length == 0)
        {
            return "Erro: Array de observações nulo ou vazio";
        }

        // Converte o array de observações para IntPtr
        IntPtr observationsPtr = ConvertArrayToPtr(observations);

        // Chama a função do Rust que processa as observações
        IntPtr resultPtr = process_mlagent_observations(observationsPtr, observations.Length);

        // Converte o resultado de volta para string
        string result = Marshal.PtrToStringAnsi(resultPtr);

        // Libera a memória alocada no Rust
        free_string(resultPtr);

        // Libera a memória do array de observações
        Marshal.FreeHGlobal(observationsPtr);

        return result;
    }

    /// <summary>
    /// Converte um array de floats para IntPtr (formato necessário para passar para Rust)
    /// </summary>
    private IntPtr ConvertArrayToPtr(float[] array)
    {
        int sizeOfFloat = sizeof(float);
        IntPtr ptr = Marshal.AllocHGlobal(array.Length * sizeOfFloat);

        for (int i = 0; i < array.Length; i++)
        {
            // Converte float para bytes e depois para int para escrever no IntPtr
            byte[] floatBytes = BitConverter.GetBytes(array[i]);
            int floatAsInt = BitConverter.ToInt32(floatBytes, 0);
            Marshal.WriteInt32(ptr, i * sizeOfFloat, floatAsInt);
        }

        return ptr;
    }

    /// <summary>
    /// Função auxiliar para converter um IntPtr em um array de floats
    /// </summary>
    private float[] IntPtrToFloatArray(IntPtr ptr, int length)
    {
        if (ptr == IntPtr.Zero || length <= 0)
        {
            return new float[0];
        }

        float[] array = new float[length];
        int sizeOfFloat = sizeof(float);

        for (int i = 0; i < length; i++)
        {
            // Lê 4 bytes (tamanho de um float) e converte de volta para float
            int floatAsInt = Marshal.ReadInt32(ptr, i * sizeOfFloat);
            byte[] floatBytes = BitConverter.GetBytes(floatAsInt);
            array[i] = BitConverter.ToSingle(floatBytes, 0);
        }

        return array;
    }

    /// <summary>
    /// Obtém ações de teste do Rust (parte 2) - valores -9 para validação
    /// </summary>
    public float[] GetTestActions(int numActions)
    {
        // Chama a função do Rust que retorna ações de teste
        FloatArray result = get_test_actions(numActions);

        // Converte o resultado para array gerenciado
        float[] actions = IntPtrToFloatArray(result.data, result.len);

        // Libera a memória alocada em Rust
        // Nota: Não podemos chamar free_float_array aqui porque Rust já moveu a posse
        // A implementação real precisaria de um mecanismo mais robusto para gerenciamento de memória

        return actions;
    }

    /// <summary>
    /// Atualiza continuamente os resultados da fila
    /// </summary>
    void Update()
    {
        // Processa resultados pendentes na fila
        while (confirmationResults.TryDequeue(out string result))
        {
            lastConfirmation = result;
        }
    }

    /// <summary>
    /// Retorna a última confirmação recebida do Rust
    /// </summary>
    public string GetLastConfirmation()
    {
        return lastConfirmation;
    }
}