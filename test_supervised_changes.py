#!/usr/bin/env python3
"""
Script de teste para verificar se as mudanças no mlagents-supervised estão funcionando.
"""
import sys
import os

# Adicionando o caminho para o módulo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml-agents/ml-agents'))

def test_imports():
    """Testar se os imports necessários funcionam"""
    print("Testando imports necessários...")
    
    try:
        from mlagents.trainers.mlagents_supervised import get_hyperparameters
        print("✓ Import da função get_hyperparameters funcionou")
    except ImportError as e:
        print(f"✗ Erro ao importar get_hyperparameters: {e}")
        return False
    
    try:
        from mlagents.trainers.ppo.optimizer_torch import PPOSettings
        from mlagents.trainers.sac.optimizer_torch import SACSettings
        from mlagents.trainers.tdsac.optimizer_torch import TDSACSettings
        from mlagents.trainers.td3.optimizer_torch import TD3Settings
        from mlagents.trainers.tqc.optimizer_torch import TQCSettings
        from mlagents.trainers.dcac.optimizer_torch import DCACSettings
        from mlagents.trainers.crossq.optimizer_torch import CrossQSettings
        from mlagents.trainers.drqv2.optimizer_torch import DrQv2Settings
        from mlagents.trainers.ppo_et.settings import PPOETSettings
        from mlagents.trainers.ppo_ce.settings import PPOCESettings
        from mlagents.trainers.sac_ae.settings import SACAESettings
        print("✓ Todos os imports de settings funcionaram")
    except ImportError as e:
        print(f"✗ Erro ao importar settings: {e}")
        return False
    
    return True

def test_get_hyperparameters():
    """Testar a função get_hyperparameters com os novos algoritmos"""
    print("\nTestando a função get_hyperparameters...")
    
    from mlagents.trainers.mlagents_supervised import get_hyperparameters
    
    algorithms = ["ppo", "sac", "tdsac", "td3", "tqc", "dcac", "crossq", "drqv2", "ppo_et", "ppo_ce", "sac_ae"]
    
    for alg in algorithms:
        try:
            settings = get_hyperparameters(alg)
            print(f"✓ get_hyperparameters('{alg}') funcionou: {type(settings).__name__}")
        except Exception as e:
            print(f"✗ get_hyperparameters('{alg}') falhou: {e}")
            return False
    
    # Testar algoritmo desconhecido (deve retornar PPOSettings por padrão)
    try:
        default_settings = get_hyperparameters("unknown_algorithm")
        from mlagents.trainers.ppo.optimizer_torch import PPOSettings
        if isinstance(default_settings, PPOSettings):
            print("✓ get_hyperparameters com algoritmo desconhecido retornou PPOSettings corretamente")
        else:
            print(f"✗ get_hyperparameters com algoritmo desconhecido retornou tipo inesperado: {type(default_settings)}")
            return False
    except Exception as e:
        print(f"✗ get_hyperparameters com algoritmo desconhecido falhou: {e}")
        return False
    
    return True

def main():
    print("Testando as mudanças no mlagents-supervised...")
    
    success = True
    
    if not test_imports():
        success = False
        
    if not test_get_hyperparameters():
        success = False
    
    if success:
        print("\n✓ Todos os testes passaram! As mudanças estão funcionando corretamente.")
        print("\nResumo das alterações:")
        print("1. Atualizado create_config.py para incluir todos os algoritmos: ppo, sac, tdsac, td3, tqc, dcac, crossq, drqv2, ppo_et, ppo_ce, sac_ae")
        print("2. Atualizado mlagents_supervised.py para suportar todos os novos algoritmos")
        print("3. Adicionados imports e lógica para suportar todos os algoritmos")
    else:
        print("\n✗ Alguns testes falharam. Verifique os erros acima.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)