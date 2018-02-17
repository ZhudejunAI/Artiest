import torch.nn as nn
import torch.optim as optim

from build_model import get_model_and_loss


def get_input_param_optimizer(input_image):
    input_param = nn.Parameter(input_image.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(content_image, style_image, input_image, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_model_and_loss(
        style_image, content_image)
    input_param, optimizer = get_input_param_optimier(input_image)

    print('Opimizing...')
    epoch = 0
    while epoch < num_epoches:

        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch += 1
            if epoch % 10 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)

    return input_param.data
