#!/usr/bin/env python3
"""
Script de teste para verificar se a correção do erro KeyError: 'hyperparameters' está funcionando.
"""
from collections import OrderedDict
from mlagents.trainers.create_rl_config import get_tdsac_params

def test_get_tdsac_params():
    print("Testando a função get_tdsac_params...")
    
    # Testar com um dicionário vazio como padrão
    defaults = OrderedDict()
    
    try:
        result = get_tdsac_params(defaults)
        print("Sucesso! A função get_tdsac_params retornou:", result)
        
        # Verificar se as chaves esperadas estão presentes
        if 'trainer_type' in result and 'hyperparameters' in result:
            print("✓ As chaves 'trainer_type' e 'hyperparameters' estão presentes no resultado")
            print(f"  trainer_type: {result['trainer_type']}")
            print(f"  hyperparameters tipo: {type(result['hyperparameters'])}")
        else:
            print("✗ Chaves ausentes no resultado")
            return False
            
        return True
        
    except KeyError as e:
        print(f"Erro de chave: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return False

if __name__ == "__main__":
    success = test_get_tdsac_params()
    if success:
        print("\n✓ Teste concluído com sucesso - o erro foi corrigido!")
    else:
        print("\n✗ Teste falhou - o erro ainda existe!")