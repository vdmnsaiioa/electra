import torch
import copy


def equivariance_check(model, train_loader):
    # Step 1: Generate a random quaternion
    rand = torch.randn(4)
    rand = rand / torch.norm(rand)

    q0, q1, q2, q3 = rand

    # Step 2: Convert quaternion to rotation matrix
    ## Remember that when using the 8 forward, we cannot compare directly every time.
    # We can only compare the final result of the density
    rotation_matrix = torch.tensor([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ], device=model.device)
    translation = torch.randn(3, device=model.device)*5
    sample_inv = next(iter(train_loader))
    sample_rot = copy.deepcopy(sample_inv)
    sample_trans = copy.deepcopy(sample_inv)
    checks = [True, True, True]
    ground_truth = sample_trans[0][0]
    print(f"Molecule: {sample_trans[0][-4].symbols}")
    if checks[0]:
        with torch.no_grad():
            fgx_eq, pred_base = model.eq_check_prediction(sample_rot, r_mat=rotation_matrix, inversion=False)
            fgx_eq_z = torch.zeros_like(fgx_eq)
            fgx_eq_z[:, :, :] = fgx_eq[:, :, :]
            eq_diff = torch.mean(torch.abs(fgx_eq_z - pred_base) / torch.min(torch.abs(fgx_eq_z) + 1e-6, torch.abs(pred_base) + 1e-6))
            eq_diff = eq_diff.cpu().numpy()
            error_rot = torch.sum(torch.abs(fgx_eq_z - ground_truth)).cpu().numpy()
            print(f"Rotation Error: {error_rot}")
    if checks[1]:
        with torch.no_grad():
            fgx_trans, pred_base = model.eq_check_prediction(sample_trans, r_mat=None, inversion=False, translation=translation)
            fgx_trans_z = torch.zeros_like(fgx_trans)
            fgx_trans_z[:, :, :] = fgx_trans[:, :, :]
            trans_diff = torch.mean(torch.abs(fgx_trans - pred_base) / torch.min(torch.abs(fgx_trans) + 1e-6, torch.abs(pred_base) + 1e-6))
            trans_diff = trans_diff.cpu().numpy()
            error_trans = torch.sum(torch.abs(fgx_trans_z - ground_truth)).cpu().numpy()
            print(f"Translation Error: {error_trans}")
    if checks[2]:
        with torch.no_grad():
            fgx_inv, pred_base = model.eq_check_prediction(sample_inv, r_mat=None, inversion=True)
            fgx_inv_z = torch.zeros_like(fgx_inv)
            fgx_inv_z[:, :, :] = fgx_inv[:, :, :]
            inv_diff = torch.mean(torch.abs(fgx_inv_z - pred_base) / torch.min(torch.abs(fgx_inv_z) + 1e-6, torch.abs(pred_base) + 1e-6))
            inv_diff = inv_diff.cpu().numpy()
            error_inv = torch.sum(torch.abs(fgx_inv_z - ground_truth)).cpu().numpy()
            print(f"Inversion Error: {error_inv}")
    eq = eq_diff < 1e-3
    inv = inv_diff < 1e-3
    trans_eq = trans_diff < 1e-3
    return (inv and eq and trans_eq)