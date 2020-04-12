from argparser import get_eval_argument, set_cuda_dev
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from src.utils import *
from src.data.voc_utils import VOC_CLASSES
from src.model import SSD

# input image normalize
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_loader = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    normalize,
    transforms.Lambda(lambda image : image.unsqueeze(0))
 ])

def detect_image(original_image, args):
    # Transform
    image = image_loader(original_image)

    # Move to default device
    image = image.cuda()

    # Forward prop
    det_results  = model(image)

    # Remove batch dimensions
    det_results = det_results[0]

    # Restore to original dimensions
    recover_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0).cuda()
    det_results[:, :4] = det_results[:, :4] * recover_dims

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    det_results = det_results.cpu()
    for i in range(det_results.size(0)):
        # Get boxes
        box_location = det_results[i][:4].tolist()
        box_label_num = int(det_results[i][5])-1
        box_label_name = VOC_CLASSES[box_label_num]
        box_color = label_color_map[box_label_name]

        # Draw
        draw.rectangle(xy=box_location, outline=box_color)
        draw.rectangle(xy=[l + 1. for l in box_location], outline=box_color)

        # Text (class label)
        text_size = font.getsize(box_label_name.upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,box_location[1]]
        draw.rectangle(xy=textbox_location, fill=box_color)
        draw.text(xy=text_location, text=box_label_name.upper(), fill='white')
    
    annotated_image.save(args.image_save_path+'p_'+args.test_image.split('/')[-1])
    del draw

if __name__ == '__main__':
    args = get_eval_argument()
    set_cuda_dev(args.ngpu)

    # Load model checkpoint
    model = SSD('test', args)
    checkpoint = args.trained_model # '*.pth'
    _, model, _ = load_checkpoint(model, args.trained_model_path+checkpoint)
    model = model.cuda()
    model.eval()
    
    with torch.no_grad():
        img_example = args.test_image # With absolute path (ex. /media/dataset/VOC2007/JPEGImages/000001.jpg)
        original_image = Image.open(img_example, mode='r')
        original_image = original_image.convert('RGB')
        detect_image(original_image, args)
        print('Detect image finished!')
