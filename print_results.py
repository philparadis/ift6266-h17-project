def show_examples():
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]


        #Image.fromarray(img_array).show()
        Image.fromarray(input).show()
        Image.fromarray(target).show()
        print i, caption_dict[cap_id]





if __name__ == '__main__':
    #resize_mscoco()
    show_examples(5, 10)
