import numpy as np

def multiplicative_decomposition(R, k=10, alpha=0.05, tolx=1e-3, maxiter=2000, bias=False):
    """
    Algoritmo NMF com atualizações multiplicativas, incluindo bias.
    Versão refatorada com melhorias na convergência e normalização.
    """
    n_drugs, n_ses = R.shape
    np.random.seed(0)

    # Inicializa apenas as partes latentes
    W_latent = np.random.rand(n_drugs, k) * 0.1
    H_latent = np.random.rand(k, n_ses) * 0.1

    if bias:
        # Inicializa biases e colunas/linhas auxiliares
        user_bias = np.random.rand(n_drugs, 1) * 0.1
        item_bias = np.random.rand(1, n_ses) * 0.1
        W = np.hstack((W_latent, user_bias, np.ones((n_drugs, 1))))
        H = np.vstack((H_latent, np.ones((1, n_ses)), item_bias))
    else:
        W, H = W_latent, H_latent

    CT = R > 0
    UN = R == 0
    eps_val = np.finfo(float).eps ** 0.5

    for iter in range(maxiter):
        W_old, H_old = np.copy(W), np.copy(H)

        # Update W
        WH = W @ H
        numer_W = (CT * R) @ H.T
        denom_W = (CT * WH + alpha * UN * WH) @ H.T + eps_val
        W = W * (numer_W / denom_W)

        # Update H
        # Re-calcula WH com o W atualizado
        WH = W @ H 
        numer_H = W.T @ (CT * R)
        denom_H = W.T @ (CT * WH + alpha * UN * WH) + eps_val
        H = H * (numer_H / denom_H)

        # Força a não-negatividade de forma explícita
        W = np.maximum(0, W)
        H = np.maximum(0, H)
        
        # Se usar bias, fixa as colunas/linhas auxiliares em 1 APÓS a atualização
        if bias:
            W[:, -1] = 1  # Última coluna de W é sempre 1
            H[-2, :] = 1 # penúltima linha de H é sempre 1

        # Normaliza APENAS a parte latente de H
        H_lat = H[:k, :]
        norm_H_lat = np.linalg.norm(H_lat, axis=1, keepdims=True)
        H[:k, :] = H_lat / (norm_H_lat + eps_val)

        # Verifica a convergência com as matrizes finais da iteração
        delta = max(
            np.max(np.abs(W - W_old) / (eps_val + np.max(np.abs(W_old)))),
            np.max(np.abs(H - H_old) / (eps_val + np.max(np.abs(H_old))))
        )
        
        if delta <= tolx:
            break
        
    R_pred=W@H
    return W, H, R_pred


def multiplicative_decomposition(R, k=10, alpha=0.05, tolx=1e-3, maxiter=2000, bias=False):
    """
    Algoritmo NMF com atualizações multiplicativas, incluindo bias.
    Versão refatorada com melhorias na convergência e normalização.
    """
    n_drugs, n_ses = R.shape
    np.random.seed(0)

    # Inicializa apenas as partes latentes
    W_latent = np.random.rand(n_drugs, k) * 0.1
    H_latent = np.random.rand(k, n_ses) * 0.1

    if bias:
        # Inicializa biases e colunas/linhas auxiliares
        user_bias = np.random.rand(n_drugs, 1) * 0.1
        item_bias = np.random.rand(1, n_ses) * 0.1
        W = np.hstack((W_latent, user_bias, np.ones((n_drugs, 1))))
        H = np.vstack((H_latent, np.ones((1, n_ses)), item_bias))
    else:
        W, H = W_latent, H_latent

    CT = R > 0
    UN = R == 0
    eps_val = np.finfo(float).eps ** 0.5

    for iter in range(maxiter):
        W_old, H_old = np.copy(W), np.copy(H)

        # Update W
        WH = W @ H
        numer_W = (CT * R) @ H.T
        denom_W = (CT * WH + alpha * UN * WH) @ H.T + eps_val
        W = W * (numer_W / denom_W)

        # Update H
        # Re-calcula WH com o W atualizado
        WH = W @ H 
        numer_H = W.T @ (CT * R)
        denom_H = W.T @ (CT * WH + alpha * UN * WH) + eps_val
        H = H * (numer_H / denom_H)

        # Força a não-negatividade de forma explícita
        W = np.maximum(0, W)
        H = np.maximum(0, H)
        
        # Se usar bias, fixa as colunas/linhas auxiliares em 1 APÓS a atualização
        if bias:
            W[:, -1] = 1  # Última coluna de W é sempre 1
            H[-2, :] = 1 # penúltima linha de H é sempre 1

        # Normaliza APENAS a parte latente de H
        H_lat = H[:k, :]
        norm_H_lat = np.linalg.norm(H_lat, axis=1, keepdims=True)
        H[:k, :] = H_lat / (norm_H_lat + eps_val)

        # Verifica a convergência com as matrizes finais da iteração
        delta = max(
            np.max(np.abs(W - W_old) / (eps_val + np.max(np.abs(W_old)))),
            np.max(np.abs(H - H_old) / (eps_val + np.max(np.abs(H_old))))
        )
        
        if delta <= tolx:
            break
        
    R_pred=W@H
    return W, H, R_pred



def PGD_decomposition(R, k=10, alpha=0.05, lambda_reg=0.02, tolx=1e-3, maxiter=2000, bias=False, learning_rate=0.001):
    n_drugs, n_ses = R.shape
    np.random.seed(0) 
    W_latent = np.random.rand(n_drugs, k) * 0.1
    H_latent = np.random.rand(k, n_ses) * 0.1
    observed_mask=(R>0)
    unobserved_mask=(R==0)

    if bias:
        mu = R[observed_mask].mean()
        print(mu)
        R_centered=observed_mask*(R-mu)
        user_bias = np.random.rand(n_drugs, 1) * 0.1
        item_bias = np.random.rand(1, n_ses) * 0.1
        W = np.hstack((W_latent, user_bias, np.ones((n_drugs, 1))))
        H = np.vstack((H_latent, np.ones((1, n_ses)), item_bias))
    else:
        mu=0.0
        R_centered=R
        W,H= W_latent,H_latent

    for iter in range(maxiter):
        
        #Update on W step
        E=W@H
        grad_W=-(observed_mask*(R_centered-E))@H.T+alpha*(unobserved_mask*E)@H.T+lambda_reg*W
        # may be a bug not sure if the shape will be cast correctly check if error
        w_step=learning_rate*grad_W
        if bias:
            W_new=W-w_step
            W_new[:,:k]=np.maximum(0,W_new[:,:k])
        else:
            W_new=np.maximum(0,W-w_step)
        if bias:
            W_new[:, -1]=1

        #Update on H step
        E_new=W_new@H
        grad_H=-W_new.T@(observed_mask*(R_centered-E_new))+alpha*W_new.T@(unobserved_mask*E_new)+lambda_reg*H
        h_step=learning_rate*grad_H
        if bias:
            H_new=H-h_step
            H_new[:k,:]=np.maximum(0,H_new[:k,:])
        else:
            H_new=np.maximum(0,H-h_step)
        if bias:
            H_new[-2,:]=1
        norms_H = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / np.where(norms_H == 0, 1, norms_H)
        H[np.isnan(H)] = 0 # Handle potential NaN after normalization


        # A very small number to prevent division by zero
        eps_val = np.finfo(float).eps

        # Step 1: Calculate the relative change for W
        # This is the norm of the difference divided by the norm of the original matrix.
        delta_W = np.linalg.norm(W_new - W, 'fro') / (np.linalg.norm(W, 'fro') + eps_val)

        # Step 2: Calculate the relative change for H
        delta_H = np.linalg.norm(H_new - H, 'fro') / (np.linalg.norm(H, 'fro') + eps_val)
        W,H=W_new,H_new

        # Step 3: Check if the maximum change is below the tolerance
        if max(delta_W, delta_H) < tolx:
            print(f"Convergence reached at iteration {iter}.") # Optional: for logging
            break # Exit the loop
    
    R_pred=W@H+mu

    return W, H, R_pred


# def als_decomposition_algorithm(R, k=10, alpha=0.05, lambda_reg=0.02, tolx=1e-3, maxiter=2000, use_bias=False):
#     """
#     Alternating Least Squares (ALS) for Non-negative Matrix Factorization (NNMF),
#     now with optional user and item biases.
#     """
#     n_drugs, n_ses = R.shape
#     np.random.seed(0) 
#     W = np.random.rand(n_drugs, k) * 0.1
#     H = np.random.rand(k, n_ses) * 0.1
    
#     # Initialize biases if use_bias is True
#     mu = 0.0
#     bu = np.zeros(n_drugs)
#     bi = np.zeros(n_ses)
#     if use_bias:
#         mu = np.mean(R[R > 0]) # Overall average of known ratings

#     # Mask for observed (R > 0) and unobserved (R == 0) entries
#     observed_mask = (R > 0)
#     unobserved_mask = (R == 0)
    
#     # Store previous W, H, bu, bi for convergence check
#     W_prev = np.copy(W)
#     H_prev = np.copy(H)
#     bu_prev = np.copy(bu)
#     bi_prev = np.copy(bi)
    
#     eps_val = np.finfo(float).eps ** 0.5

#     for iter_count in range(maxiter):
#         if iter_count+1 % 5 == 0:
#             print(f'iter_count: {iter_count+1}')
#         # 1. Update bu (user biases) - only if use_bias is True
#         if use_bias:
#             for u_idx in range(n_drugs):
#                 observed_cols = np.where(observed_mask[u_idx, :])[0]
#                 if len(observed_cols) > 0:
#                     current_prediction_base = W[u_idx, :] @ H[:, observed_cols]
#                     residual_sum = np.sum(R[u_idx, observed_cols] - (mu + bi[observed_cols] + current_prediction_base))
#                     bu[u_idx] = residual_sum / (len(observed_cols) + lambda_reg)
#                 else:
#                     bu[u_idx] = 0

#             # 2. Update bi (item biases) - only if use_bias is True
#             for i_idx in range(n_ses):
#                 observed_rows = np.where(observed_mask[:, i_idx])[0]
#                 if len(observed_rows) > 0:
#                     current_prediction_base = W[observed_rows, :] @ H[:, i_idx]
#                     residual_sum = np.sum(R[observed_rows, i_idx] - (mu + bu[observed_rows] + current_prediction_base))
#                     bi[i_idx] = residual_sum / (len(observed_rows) + lambda_reg)
#                 else:
#                     bi[i_idx] = 0

#         # 3. Update W (fix H, mu, bu, bi)
#         for i in range(n_drugs):
#             # Target for NNLS: R_effective = R_ui - (mu + bu_u + bi_i) for observed entries.
#             # If not using bias, R_effective is just R_ui.
            
#             current_row_observed_idx = np.where(observed_mask[i, :])[0]
#             current_row_unobserved_idx = np.where(unobserved_mask[i, :])[0]

#             # For observed entries: A = H_T, b = R_corrected_obs
#             R_corrected_obs = R[i, current_row_observed_idx] - (mu + bu[i] + bi[current_row_observed_idx]) if use_bias else R[i, current_row_observed_idx]
#             A_obs = H[:, current_row_observed_idx].T
            
#             # For unobserved entries: A = sqrt(alpha) * H_T, b = 0
#             A_unobs = H[:, current_row_unobserved_idx].T
            
#             # Add regularization for W[i,:] if biases are used
#             A_reg_rows = np.sqrt(lambda_reg) * np.eye(k) if use_bias else np.array([]) # Empty if not using bias
#             b_reg_rows = np.zeros(k) if use_bias else np.array([]) # Empty if not using bias

#             # Combine all parts for NNLS
#             A_combined = np.vstack([A_obs, np.sqrt(alpha) * A_unobs, A_reg_rows]) if use_bias else np.vstack([A_obs, np.sqrt(alpha) * A_unobs])
#             b_combined = np.hstack([R_corrected_obs, np.zeros(len(current_row_unobserved_idx)), b_reg_rows]) if use_bias else np.hstack([R_corrected_obs, np.zeros(len(current_row_unobserved_idx))])
            
#             # Handle potential empty arrays if no observed/unobserved data for a row or no bias
#             if A_combined.shape[0] > 0 and A_combined.shape[1] > 0:
#                 W[i, :], _ = nnls(A_combined, b_combined)
#             else: 
#                 W[i, :] = np.zeros(k) 

#         # 4. Update H (fix W, mu, bu, bi)
#         for j in range(n_ses):
#             current_col_observed_idx = np.where(observed_mask[:, j])[0]
#             current_col_unobserved_idx = np.where(unobserved_mask[:, j])[0]

#             # For observed entries: A = W, b = R_corrected_obs
#             R_corrected_obs = R[current_col_observed_idx, j] - (mu + bu[current_col_observed_idx] + bi[j]) if use_bias else R[current_col_observed_idx, j]
#             A_obs = W[current_col_observed_idx, :]
            
#             # For unobserved entries: A = sqrt(alpha) * W, b = 0
#             A_unobs = W[current_col_unobserved_idx, :]

#             # Add regularization for H[:,j] if biases are used
#             A_reg_cols = np.sqrt(lambda_reg) * np.eye(k) if use_bias else np.array([])
#             b_reg_cols = np.zeros(k) if use_bias else np.array([])

#             # Combine all parts for NNLS (for H update)
#             A_combined = np.vstack([A_obs, np.sqrt(alpha) * A_unobs, A_reg_cols]) if use_bias else np.vstack([A_obs, np.sqrt(alpha) * A_unobs])
#             b_combined = np.hstack([R_corrected_obs, np.zeros(len(current_col_unobserved_idx)), b_reg_cols]) if use_bias else np.hstack([R_corrected_obs, np.zeros(len(current_col_unobserved_idx))])

#             if A_combined.shape[0] > 0 and A_combined.shape[1] > 0:
#                 H[:, j], _ = nnls(A_combined, b_combined)
#             else: 
#                 H[:, j] = np.zeros(k) 

#         # Normalize H (rows of H) as in the paper
#         norms_H = np.linalg.norm(H, axis=1, keepdims=True)
#         H = H / np.where(norms_H == 0, 1, norms_H)
#         H[np.isnan(H)] = 0 # Handle potential NaN after normalization

#         # Check for convergence (considering W, H, and biases if used)
#         delta_W = np.max(np.abs(W - W_prev) / (eps_val + np.max(np.abs(W_prev))))
#         delta_H = np.max(np.abs(H - H_prev) / (eps_val + np.max(np.abs(H_prev))))
        
#         print(f'delta_W: {delta_W}')
#         print(f'delta_H: {delta_H}')
#         if use_bias:
#             delta_bu = np.max(np.abs(bu - bu_prev) / (eps_val + np.max(np.abs(bu_prev))))
#             delta_bi = np.max(np.abs(bi - bi_prev) / (eps_val + np.max(np.abs(bi_prev))))
#             convergence_delta = max(delta_W, delta_H, delta_bu, delta_bi)
#         else:
#             convergence_delta = max(delta_W, delta_H)
        
#         if convergence_delta <= tolx:
#             break
            
#     # Final prediction: WH + biases if used, else just WH
#     R_pred = W @ H
#     if use_bias:
#         R_pred += mu + bu[:, np.newaxis] + bi[np.newaxis, :]
        
#     return W, H, mu, bu, bi, R_pred