#!/usr/bin/env python

"""
Script de teste para verificar se a configuração de treinamento supervisionado
é lida e interpretada corretamente pelo sistema de configuração do ML-Agents.
"""

import yaml
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.cli_utils import load_config


def test_supervised_config_loading():
    """
    Testa o carregamento da configuração supervisionada
    """
    print("Testando carregamento da configuração supervisionada...")
    
    try:
        # Carregar configuração de exemplo
        config_path = "test_supervised_config.yaml"
        config_dict = load_config(config_path)
        
        print("Configuração carregada com sucesso!")
        print(f"Comportamentos definidos: {list(config_dict['behaviors'].keys())}")
        
        # Verificar se a seção supervisionada foi carregada
        behavior_name = list(config_dict['behaviors'].keys())[0]
        behavior_config = config_dict['behaviors'][behavior_name]
        
        if 'supervised' in behavior_config and behavior_config['supervised'] is not None:
            print(f"Seção supervisionada encontrada para o comportamento '{behavior_name}':")
            supervised_config = behavior_config['supervised']
            print(f"  - CSV Path: {supervised_config.get('csv_path')}")
            print(f"  - Observation Columns: {supervised_config.get('observation_columns')}")
            print(f"  - Action Columns: {supervised_config.get('action_columns')}")
            print(f"  - Num Epochs: {supervised_config.get('num_epoch')}")
            print(f"  - Batch Size: {supervised_config.get('batch_size')}")
            print("  - Configuração supervisionada carregada corretamente!")
        else:
            print(f"Nenhuma seção supervisionada encontrada para '{behavior_name}'")
        
        # Converter para RunOptions para testar estruturação completa
        run_options = RunOptions.from_dict(config_dict)
        
        print("\nConfiguração convertida para RunOptions com sucesso!")
        
        # Verificar se as configurações supervisionadas estão presentes
        for name, trainer_settings in run_options.behaviors.items():
            if trainer_settings.supervised is not None:
                print(f"\nConfigurações supervisionadas estruturadas para '{name}':")
                supervised = trainer_settings.supervised
                print(f"  - CSV Path: {supervised.csv_path}")
                print(f"  - Observation Columns: {supervised.observation_columns}")
                print(f"  - Action Columns: {supervised.action_columns}")
                print(f"  - Num Epochs: {supervised.num_epoch}")
                print(f"  - Early Stopping: {supervised.early_stopping}")
        
        print("\n✅ Teste de configuração supervisionada concluído com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro no teste de configuração supervisionada: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_supervised_config_loading()