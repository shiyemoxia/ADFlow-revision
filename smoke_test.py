"""Fast server-side checks that do not load datasets or pretrained weights."""

import torch

from snet_loss import adflow_training_loss, get_logp


def check_distribution(distribution, device):
    batch_size, feature_dim = 16, 8
    z = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
    logdet = torch.randn(batch_size, device=device, requires_grad=True)
    mask = torch.randint(0, 2, (batch_size, 1), device=device).float()

    if distribution == 'adaptive':
        variances = torch.tensor([0.5, 1.0, 2.0], device=device, requires_grad=True)
        weights = torch.softmax(torch.randn(3, device=device, requires_grad=True), dim=0)
    else:
        variances = weights = None

    log_prob = get_logp(
        feature_dim,
        z,
        logdet,
        variances,
        weights,
        distribution=distribution,
    )
    loss, nll, bce = adflow_training_loss(
        feature_dim,
        z,
        logdet,
        mask,
        variances,
        weights,
        distribution=distribution,
    )
    assert log_prob.shape == (batch_size,)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(loss)
    loss.backward()
    print('{}: loss={:.5f}, nll={:.5f}, bce={:.5f}'.format(
        distribution, float(loss.detach()), float(nll), float(bce)
    ))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('torch={}'.format(torch.__version__))
    print('device={}'.format(device))
    if device.type == 'cuda':
        print('gpu={}'.format(torch.cuda.get_device_name(0)))
    check_distribution('standard', device)
    check_distribution('adaptive', device)
    print('ADFlow smoke test passed')


if __name__ == '__main__':
    main()
