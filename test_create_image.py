from PIL import Image
img = Image.new('RGB', (224, 224), color=(180, 140, 120))
img.save(r'D:\visual\test_img.png')
print('Dummy image created')
