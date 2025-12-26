use std::ffi::CString;
use std::os::raw::c_char;

// Estrutura para gerenciar arrays de floats (para observações do ML-Agents)
#[repr(C)]
pub struct FloatArray {
    pub data: *mut f32,
    pub len: usize,
}

/// Função que recebe observações do ML-Agent (array de floats) e retorna uma string de confirmação
/// Esta função será chamada pelo C# quando o método CollectObservations for interceptado
#[no_mangle]
pub extern "C" fn process_mlagent_observations(observations: *const f32, length: usize) -> *mut c_char {
    if observations.is_null() || length == 0 {
        let error_msg = "Erro: Array de observações nulo ou vazio";
        let c_string = CString::new(error_msg).expect("Falha ao converter string Rust para string C");
        return c_string.into_raw();
    }

    // Converter o ponteiro para slice para poder acessar os dados
    let _obs_slice = unsafe { std::slice::from_raw_parts(observations, length) };

    // Criar a string de confirmação com o tamanho das observações
    let confirmation_msg = format!("Observacoes recebidas, o tamanho das observacoes e: {}", length);

    // Converter para CString e retornar o ponteiro bruto
    let c_string = CString::new(confirmation_msg).expect("Falha ao converter string Rust para string C");
    c_string.into_raw()  // O C# será responsável por liberar a memória com free_string
}

/// Função para liberar a memória alocada para strings
#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

/// Função para liberar a memória alocada para arrays de floats
#[no_mangle]
pub extern "C" fn free_float_array(data: *mut f32, len: usize) {
    if !data.is_null() && len > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(data, len, len);
        }
    }
}

/// Função temporária para testar a integração
/// Retorna um array de ações com valores -9 como solicitado para a parte 2
#[no_mangle]
pub extern "C" fn get_test_actions(num_actions: usize) -> FloatArray {
    if num_actions == 0 {
        return FloatArray {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }

    // Criar um vetor com valores -9 para todas as ações
    let actions: Vec<f32> = vec![-9.0; num_actions];

    // Converter para caixa e extrair o ponteiro
    let boxed_actions = actions.into_boxed_slice();
    let len = boxed_actions.len();
    let ptr = Box::into_raw(boxed_actions) as *mut f32;

    FloatArray { data: ptr, len }
}

/// Função para obter dados do array de floats (para debugging)
#[no_mangle]
pub extern "C" fn get_float_array_data(array: FloatArray, index: usize) -> f32 {
    if array.data.is_null() || index >= array.len {
        return 0.0; // valor padrão para índice inválido
    }

    unsafe {
        let slice = std::slice::from_raw_parts(array.data, array.len);
        slice[index]
    }
}