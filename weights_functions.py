        print("Treinamento supervisionado concluído.")
        
        # Salvar pesos treinados para verificação posterior
        self._save_trained_weights(artifact_path)
        
        return metrics
    
    def _save_trained_weights(self, artifact_path: Optional[str]) -> None:
        """
        Salva os pesos treinados para verificação posterior.
        :param artifact_path: Caminho para salvar os pesos
        """
        if artifact_path:
            try:
                import os
                weights_path = f"{artifact_path}/supervised_weights.pth"
                # Criar diretório se não existir
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                torch_module.save(self.policy.actor.state_dict(), weights_path)
                print(f"[AUDIT] Pesos supervisionados salvos: {weights_path}")
                # Calcular e registrar hash dos pesos para verificação
                weights_hash = self._calculate_weights_hash()
                print(f"[AUDIT] Hash dos pesos supervisionados: {weights_hash}")
            except Exception as e:
                print(f"Erro ao salvar pesos supervisionados: {e}")
    
    def _calculate_weights_hash(self) -> str:
        """
        Calcula um hash dos pesos atuais para verificação.
        :return: Hash MD5 dos pesos atuais
        """
        try:
            import hashlib
            import pickle
            
            # Serializar os pesos atuais
            state_dict = self.policy.actor.state_dict()
            # Ordenar as chaves para garantir consistência
            sorted_items = sorted(state_dict.items())
            weights_bytes = pickle.dumps(sorted_items)
            # Calcular hash
            return hashlib.md5(weights_bytes).hexdigest()
        except Exception as e:
            print(f"Erro ao calcular hash dos pesos: {e}")
            return "unknown"
    
    def get_trained_weights_hash(self) -> str:
        """
        Retorna o hash dos pesos treinados para verificação.
        :return: Hash MD5 dos pesos treinados
        """
        return self._calculate_weights_hash()
    
    def compare_weights_with_saved(self, weights_file_path: str) -> bool:
        """
        Compara os pesos atuais com pesos salvos em arquivo.
        :param weights_file_path: Caminho para arquivo de pesos
        :return: True se os pesos são iguais, False caso contrário
        """
        try:
            import os
            if not os.path.exists(weights_file_path):
                print(f"Arquivo de pesos não encontrado: {weights_file_path}")
                return False
                
            current_state = self.policy.actor.state_dict()
            saved_state = torch_module.load(weights_file_path)
            
            # Comparar cada tensor de peso
            for key in current_state.keys():
                if key not in saved_state:
                    print(f"Chave {key} não encontrada nos pesos salvos")
                    return False
                    
                if not torch_module.equal(current_state[key], saved_state[key]):
                    print(f"Pesos diferentes na chave {key}")
                    return False
                    
            print("Pesos verificados: IGUAIS")
            return True
            
        except Exception as e:
            print(f"Erro ao comparar pesos: {e}")
            return False