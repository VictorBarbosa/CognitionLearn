#!/usr/bin/env python3
"""
Script de teste para verificar se a função _dict_to_str está filtrando corretamente valores None.
"""

import sys
import os

# Adicionar os diretórios do projeto ao path
project_root = os.path.dirname(__file__)
mlagents_path = os.path.join(project_root, 'ml-agents')
mlagents_envs_path = os.path.join(project_root, 'ml-agents-envs')

# Adicionar ambos ao path
sys.path.insert(0, mlagents_path)
sys.path.insert(0, mlagents_envs_path)

def test_dict_to_str():
    """Testa a função _dict_to_str com valores None."""
    try:
        # Importar a função após adicionar o path
        from mlagents.trainers.stats import _dict_to_str
        
        # Testar com um dicionário que contém valores None
        test_dict = {
            "ppo": None,
            "sac": None,
            "td3": None,
            "tdsac": None,
            "tqc": None,
            "poca": None,
            "drqv2": None,
            "dcac": None,
            "crossq": None,
            "ppo_et": None,
            "ppo_ce": None,
            "sac_ae": None,
            "supervised": None,
            "trainer_type": "all",
            "hyperparameters": {
                "batch_size": 1024,
                "buffer_size": 10240,
                "learning_rate": 0.0003,
                "checkpoint_interval": 5000
            },
            "network_settings": {
                "normalize": True,
                "hidden_units": 512,
                "num_layers": 2,
                "memory": None
            }
        }
        
        print("Testando _dict_to_str com valores None...")
        result = _dict_to_str(test_dict, 0)
        print("Resultado:")
        print(result)
        
        # Verificar se os valores None foram filtrados
        if "ppo: None" in result or "sac: None" in result:
            print("\n❌ Falha no teste: Valores None ainda estão aparecendo!")
            return False
        else:
            print("\n✅ Sucesso no teste: Valores None foram filtrados corretamente!")
            return True
            
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dict_to_str()
    if success:
        print("\n✅ Todos os testes passaram!")
    else:
        print("\n❌ Alguns testes falharam!")
    sys.exit(0 if success else 1)