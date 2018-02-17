from torch.autograd import Variable
from torchvision import transforms
from run_artiest import run_style_transfer
from load_image import load_img, show_img

style_image = load_img('./images/style3.jpg')
style_image = Variable(style_image)
content_image = load_img('./images/dog.jpg')
content_image = Variable(content_image)

input_image = content_image.clone()

out = run_style_transfer(content_image, style_image, input_image, num_epoches=300)

show_img(out.cpu())

save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))

print(save_pic)

save_pic.save('./picture/picture.jpg')
