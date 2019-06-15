%Final report of Robotic vision. 
%Written by: Jui-Wen, Yeh 
%The target of the program is to make a code that can automatically choose
%and do the focus protraits. 
%first step is to input the picture, take a look at it. 
%Use Viola-Jones method. 
%ref by: https://www.youtube.com/watch?v=JXrjp2BXgn8
%ref by: https://www.wikiwand.com/en/Viola%E2%80%93Jones_object_detection_framework
%ref by: https://www.vocal.com/video/face-detection-using-viola-jones-algorithm/
%As do not have equipment that is good enough, didn't train own module. 
%how to train module: https://www.mathworks.com/help/vision/ug/train-a-cascade-object-detector.html
%concept of the code is to use Viola-Jones object detection with
%pre-trained face recognition and nose recognition. use nose recognition to
%be the filter to check if it's a face or not. 
%clear all, clc
cd .., cd training_data\
pic_case = imread('image_0300.jpg'); 
temp_pic_original = pic_case; 
cd ..
mkdir result, cd result
imwrite(pic_case, 'original_pic.png');
mkdir face_detection, mkdir threshold, cd face_detection
mkdir k-cluster, mkdir circle_method
cd circle_method 
mkdir attempt_1_with_final_filter, mkdir attempt_2_with_adjustion
cd .., cd .., cd threshold
gray_pic = rgb2gray(pic_case); 
[size_pic_x, size_pic_y] = size(gray_pic); 
R_shell_pic = uint8(pic_case(:, :, 1));
G_shell_pic = uint8(pic_case(:, :, 2));
B_shell_pic = uint8(pic_case(:, :, 3)); 
% compare colors
%figure(1)
subplot (3, 2, 1)
imshow(pic_case)
title ('original')
subplot (3, 2, 2)
imshow(gray_pic)
title ('gray picture')
subplot (3, 2, 3)
imshow(R_shell_pic)
title ('R sub')
subplot (3, 2, 4)
imshow(G_shell_pic)
title ('G sub')
subplot (3, 2, 5)
imshow(B_shell_pic)
title ('b sub')
%print('-f1','color_seperation','-dpng')

%threshold 
%threshold should be between 0-256
output = zeros(size_pic_x, size_pic_y);
threshold = 50; 
%gray picture binarize 
for x = 1: size_pic_x
    for y = 1: size_pic_y
        temp_input = gray_pic(x, y); 
        if temp_input > threshold
            output(x, y) = 256; 
        else 
            output(x, y) = 0; 
        end 
    end 
end 
%gray image thresed 

%figure (2) 
imshow (output)
title ('gray with threshold')
%print('-f2','gray_threshold','-dpng')

%three color input
%skin detection 
input= zeros(size_pic_x, size_pic_y, 3); 
output_RGB= zeros(size_pic_x, size_pic_y, 3); 
input(:, :, 1) = R_shell_pic; 
input(:, :, 2) = G_shell_pic;
input(:, :, 3) = B_shell_pic;
R_thres_LOW = 219; 
G_thres_LOW = 144; 
B_thres_LOW = 101; 
R_thres_HIGH = 259; 
G_thres_HIGH = 233; 
B_thres_HIGH = 196;
threshold_LOW_RGB = zeros(3); 
threshold_HIGH_RGB = zeros(3); 
threshold_LOW_RGB(1) = R_thres_LOW; 
threshold_LOW_RGB(2) = G_thres_LOW; 
threshold_LOW_RGB(3) = B_thres_LOW;  
threshold_HIGH_RGB(1) = R_thres_HIGH; 
threshold_HIGH_RGB(2) = G_thres_HIGH; 
threshold_HIGH_RGB(3) = B_thres_HIGH; 

for i = 1:3
    for x = 1: size_pic_x
        for y = 1: size_pic_y
            temp_input = input(x, y, i); 
            if temp_input > threshold_LOW_RGB(i) && temp_input < threshold_HIGH_RGB(i)
                output_RGB(x, y, i) = 256; 
            else 
                output_RGB(x, y, i) = 0; 
            end 
        end 
    end 
end


no_bin = im2bw(output_RGB, 0.5);
acht_bin = im2bw(output_RGB, 0.8);
%figure (3)
subplot(1, 3 ,1)
imshow(output_RGB)
title('threshold')
subplot(1, 2, 2)
imshow(no_bin)
title('binary bounded')
%print('-f3','threshold_vs_boundary','-dpng')

Lab_pic_original = rgb2lab(pic_case); 
Lab_pic_output = rgb2lab(output_RGB); 
%Use image segmentation k-means to cluster the objects into three clusters.
abs_original = Lab_pic_original(:, :, 2:3); 
abs_original = im2single(abs_original); 
abs_output = Lab_pic_output(:, :, 2:3); 
abs_output = im2single(abs_output);
nColors = 4; 
pixel_labels_original = imsegkmeans(abs_original,nColors,'NumAttempts',3);
pixel_labels_output = imsegkmeans(abs_output,nColors,'NumAttempts',3);
%figure(4)
subplot(1, 3, 1)
imshow(pic_case)
title('original picture'); 
subplot(1, 3, 2)
imshow(pixel_labels_original,[])
title('Image Labeled by Cluster Index--original');
subplot(1, 3, 3)
imshow(pixel_labels_output,[])
title('Image Labeled by Cluster Index--output');
%print('-f4','original_vs_several_cluster','-dpng')

pix_num = zeros(3); 
LAB_original_output = zeros(size_pic_x, size_pic_y); 
for i = 1:size_pic_x
    for j = 1:size_pic_y
        if pixel_labels_original(i, j) ==1
            pix_num(1) = pix_num(1)+1; 
            LAB_original_output (i, j)= 0; 
        elseif pixel_labels_original(i, j) ==2
            pix_num(2) = pix_num(2)+1; 
            LAB_original_output (i, j)= 64; 
        else 
            pix_num(3) = pix_num(3)+1; 
            LAB_original_output (i, j)= 128;   
        end 
    end 
end 
%figure (5)
imshow (~LAB_original_output, [])
%print('-f5','LAB_color','-dpng')

cd ..
cd face_detection
%write center into the for loop and pin it 
%take advantage of the rectangle to find the center of the face. 
Face_detect = vision.CascadeObjectDetector; 
BD_detect = step(Face_detect, pic_case); 
center_face = zeros(size(BD_detect, 1), 2); 
for i = 1: size(BD_detect, 1) 
    center_face(i, 1) = BD_detect(i, 1)+BD_detect(i, 3)/2; 
    center_face(i, 2) = BD_detect(i, 2)+BD_detect(i, 4)/2;     
end
sobel_bw = edge(gray_pic, 'approxcanny');
%figure(6)
imshow(sobel_bw); hold on 
for i = 1: size(BD_detect, 1)
    rectangle ('position', BD_detect(i, :), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
    plot(center_face(i, 1), center_face(i, 2),'.r','MarkerSize',30);
end 
title ('used trained cascade to detect face, example 1'); 
hold off
%print('-f6','find_face','-dpng')


%detect nose to confirm it's a humanbeing
Nose_Detect = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 16); 
BD_NS = step(Nose_Detect, pic_case); 
center_nose = zeros(size(BD_NS, 1), 2); 
for i = 1: size(BD_NS, 1) 
    center_nose(i, 1) = BD_NS(i, 1)+BD_NS(i, 3)/2; 
    center_nose(i, 2) = BD_NS(i, 2)+BD_NS(i, 4)/2;     
end

%put the nose position onto the graph
for i = 1: size(BD_NS, 1)
    rectangle ('position', BD_NS(i, :), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'b'); 
    plot(center_nose(i, 1), center_nose(i, 2),'.b','MarkerSize',30);
end 
title ('test for nose detecting, example 1'); 
hold off

%as nose will definitly be in the range of the face, one can confirm the face 
%with it. 
for i = 1: size(center_nose, 1)
    for j = 1: size(BD_detect, 1)
        if center_nose(i, 1) > BD_detect(j , 1) && center_nose(i, 1) < BD_detect(j , 1)+ BD_detect(j, 3)
            if center_nose(i, 2) > BD_detect(j , 2) && center_nose(i, 2) < BD_detect(j , 2)+ BD_detect(j, 4)
                actual_face(i, :) = BD_detect(j, :); 
            end 
        end 
    end 
end 

%figure(7)
imshow(sobel_bw); hold on 
for i = 1: size(BD_NS, 1)
    rectangle ('position', actual_face(i, :), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'g'); 
    plot(center_nose(i, 1), center_nose(i, 2),'.b','MarkerSize',30);
end 
title ('confirm real face detecting, example 1'); 
hold off
%print('-f7','find_nose','-dpng')

%////////////////////
%with the method above, one can detect the real face and the center of the face. 
%The next step is to blur it and adjust it. 
cd k-cluster
%Normally for a background substraction, one should have the reference
%background. But as we are doing this with a single picture, we will try to
%use color based segmentation to substract it. s

%figure(8)
imshow(pic_case)
%print('-f8','original_figure','-dpng')

LAB_pic = rgb2lab(pic_case); 
ab = LAB_pic(:,:,2:3);
ab = im2single(ab);
nColors = 4;
% repeat the clustering 4 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
%figure(9)
imshow(pixel_labels,[])
title('Image Labeled by Cluster Index');
%print('-f9','color_cluster','-dpng')

%seperate different color  #1
mask1 = pixel_labels==1;
cluster1 = pic_case .* uint8(mask1);
%seperate different color  #2
mask2 = pixel_labels==2;
cluster2 = pic_case .* uint8(mask2);
%seperate different color  #3
mask3 = pixel_labels==3;
cluster3 = pic_case .* uint8(mask3);
%seperate different color  #4
mask4 = pixel_labels==4;
cluster4 = pic_case .* uint8(mask4);


%figure(10)
imshow(cluster1)
title('Objects in Cluster 1');
%print('-f10','cluster1','-dpng')
%figure(11)
imshow(cluster2)
title('Objects in Cluster 2');
%print('-f11','cluster2','-dpng')
%figure(12)
imshow(cluster3)
title('Objects in Cluster 3');
%print('-f12','cluster3','-dpng')
%figure(13)
imshow(cluster4)
title('Objects in Cluster 4');
%print('-f13','cluster4','-dpng')



%reach out one: 
%so the method is to use annotation to find which color has the largest
%number in the region of the face. Set it as the main cluster, blur the 
%other ones. 
%the fuction of the flitered real face is called 'actual_face'
counter = zeros(1, nColors); 
[actual_size_x, actual_size_y] = size(actual_face); 
for i = 1: actual_size_x
    for x_dir = actual_face(i, 1): actual_face(i, 1)+actual_face(i, 3)-1
        for y_dir = actual_face(i, 2): actual_face(i, 2)+actual_face(i, 4)-1
            if pixel_labels(y_dir, x_dir)==1
                counter(1) = counter(1)+1; 
            elseif pixel_labels(y_dir, x_dir)==2
                counter(2) = counter(2)+1; 
            elseif pixel_labels(y_dir, x_dir)==2
                counter(3) = counter(3)+1; 
            else 
                counter(4) = counter(4)+1; 
            end 
        end
    end
end
%annotation result winner 
anno_no1 = find(counter == max(counter(:)));
counter(anno_no1) = 0; 
anno_no2 = find(counter == max(counter(:)));
counter(anno_no2) = 0; 
anno_no3 = find(counter == max(counter(:)));
counter(anno_no3) = 0;
temp_counter = [1:4]; 
temp_counter(find(temp_counter==anno_no1)) = 0; 
temp_counter(find(temp_counter==anno_no2)) = 0; 
temp_counter(find(temp_counter==anno_no3)) = 0; 
anno_no4 = find(temp_counter == max(temp_counter(:)));

%draw the result 
formatspec='cluster%d'; 
formatspec_anno='anno_no%d';
true_res = uint8(zeros(size_pic_x, size_pic_y)); 
for i = 1: nColors
    temp_anno = sprintf(formatspec_anno,i); 
    temp_anno = eval(temp_anno); 
    temp_max_num = temp_anno; 
    temp = sprintf(formatspec,temp_max_num);
    temp_max = eval (temp); 
    if i == 1 
        temp_res = temp_max; 
    else
        H = fspecial('disk',(i*1.2-1)*(i*0.7));
        temp_res = imfilter(temp_max,H,'replicate');  
    end 
    true_res = true_res + temp_res;    
end   
%figure(14)
subplot(2, 2, 1)
imshow(pic_case); 
title('before processing ')
subplot(2, 2, 2)
imshow(true_res); 
title('after processing ')
subplot(2, 2, 3)
imshow(histeq(true_res)); 
title('equalized processing ')
hold on 
sig = 0.5;  
hsize = 3 * sig * 2 +1; 
h = fspecial('gaussian', hsize, sig);
Gaussian_res = imfilter(true_res, h);
subplot(2, 2, 4)
imshow(Gaussian_res); 
title('Gaussian processing ')
%print('-f14','cluster_and_some_cluster_twist','-dpng')
cd ..

%reach 2, via KLT algorithm, one can find the eigenpoints of the face. 
%Use it as the edge, draw several circles to blurr it with different 
%magnitutde. 
cd circle_method
%ref: https://www.mathworks.com/help/vision/examples/face-detection-and-tracking-using-the-klt-algorithm.html
points = detectMinEigenFeatures(rgb2gray(pic_case), 'ROI', BD_detect);
% Display the detected points.
%figure(15), imshow(pic_case), hold on, title('Detected features');
plot(points);
position = points.Location; 
[pt_number, temp_num] = size(position); 
counter = zeros(pt_number, 1); 
for i = 1: pt_number
    counter(i) = abs(center_face(1)-position(i, 1)).^2+abs(center_face(2)-position(i, 2)).^2;
end
%print('-f15','KLT_specialized_detection','-dpng')

%filter weird points
for i = 1:20
    max_dis_sqrt = max(counter(:));
    [s, sr] = find(counter == max_dis_sqrt);  
    counter(s, sr) = 0; 
end 

max_dis_sqrt = max(counter(:));
dis_pic_sqrt = zeros(size_pic_x, size_pic_y);
for x = 1: size_pic_x
    for y = 1: size_pic_y
        dis_pic_sqrt(x, y) = abs(center_face(2)-x).^2+abs(center_face(1)-y).^2;     
    end 
end
%figure (16)
mesh(dis_pic_sqrt)
title('find the center of picture')
%print('-f16','pic_face_center_KLT','-dpng')

%threshold, adjust the threshold by distance. the further, the more blurred
%it is. But the distance within the upper body area should be not blurred. 
% put the threshold, set the amount of threshold is 10. 
threshold_pic = zeros(size_pic_x, size_pic_y); 
threshold_num = 10; 
processing_pic = zeros(size_pic_x, size_pic_y, threshold_num);
%_____________________________________________________
%adjust threshold here
t_thres1 = 6e+3; 
t_thres2 = 0.8e+5; 
t_thres3 = 1.5e+5; 
t_thres4 = 2e+5; 
t_thres5 = 3.5e+5; 
%_____________________________________________________
th1 = int32(max_dis_sqrt .* 0.001 + t_thres1)/2; 
th2 = t_thres1; 
th3 = int32(t_thres1 + t_thres2)/2; 
th4 = t_thres2; 
th5 = int32(t_thres2 + t_thres3)/2; 
th6 = t_thres3; 
th7 = int32(t_thres3 + t_thres4)/2; 
th8 = t_thres4; 
th9 = int32(t_thres4 + t_thres5)/2; 

after_thres1 = zeros(size_pic_x, size_pic_y, 3);
after_thres2 = zeros(size_pic_x, size_pic_y, 3);
after_thres3 = zeros(size_pic_x, size_pic_y, 3);
after_thres4 = zeros(size_pic_x, size_pic_y, 3);
after_thres5 = zeros(size_pic_x, size_pic_y, 3);
after_thres6 = zeros(size_pic_x, size_pic_y, 3);
after_thres7 = zeros(size_pic_x, size_pic_y, 3);
after_thres8 = zeros(size_pic_x, size_pic_y, 3);
after_thres9 = zeros(size_pic_x, size_pic_y, 3);
after_thres10 = zeros(size_pic_x, size_pic_y, 3);

%Do the threshold here
for t = 1: 3
    for x = 1: size_pic_x
        for y = 1: size_pic_y
            if dis_pic_sqrt(x, y) < max_dis_sqrt
                threshold_pic(x, y) = 0; 
                processing_pic(x, y, 1) =  pic_case(x, y);
                after_thres1(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) >= max_dis_sqrt && dis_pic_sqrt(x, y) < th1
                threshold_pic(x, y) = 1; 
                processing_pic(x, y, 2) = pic_case(x, y);
                after_thres2(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th1 && dis_pic_sqrt(x, y) < th2
                threshold_pic(x, y) = 2; 
                processing_pic(x, y, 3) =  pic_case(x, y);
                after_thres3(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th2 && dis_pic_sqrt(x, y) < th3
                threshold_pic(x, y) = 3; 
                processing_pic(x, y, 4) =  pic_case(x, y);
                after_thres4(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th3 && dis_pic_sqrt(x, y) < th4
                threshold_pic(x, y) = 4;  
                processing_pic(x, y, 5) =  pic_case(x, y);
                after_thres5(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th4 && dis_pic_sqrt(x, y) < th5
                threshold_pic(x, y) = 5;  
                processing_pic(x, y, 6) =  pic_case(x, y);
                after_thres6(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th5 && dis_pic_sqrt(x, y) < th6
                threshold_pic(x, y) = 6;  
                processing_pic(x, y, 7) =  pic_case(x, y);
                after_thres7(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th6 && dis_pic_sqrt(x, y) < th7
                threshold_pic(x, y) = 7;   
                processing_pic(x, y, 8) =  pic_case(x, y);
                after_thres8(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th7 && dis_pic_sqrt(x, y) < th8
                threshold_pic(x, y) = 8;    
                processing_pic(x, y, 9) =  pic_case(x, y);
                after_thres9(x, y, t) =  pic_case(x, y, t);
            elseif dis_pic_sqrt(x, y) > th8 
                threshold_pic(x, y) = 9;  
                processing_pic(x, y, 10) =  pic_case(x, y);
                after_thres10(x, y, t) =  pic_case(x, y, t);
            end        
        end 
    end 
end
%figure(17)
mesh(threshold_pic)
title('threshold_pic')
%print('-f17','thresold_3D','-dpng')

%for debugging issue 
temp_pic = zeros(size_pic_x, size_pic_y); 
temp_pic_eins = zeros(size_pic_x, size_pic_y, 3); 
for i = 1: 10
    temp_pic = temp_pic + processing_pic(:, :, i); 
end 
temp_pic_eins = temp_pic_eins+ after_thres1; 
temp_pic_eins = temp_pic_eins ./ 256; 
%figure(18)
mesh(temp_pic)
title('temp_pic')
%print('-f18','pic_KLM_3D','-dpng')

%figure(19)
imshow(temp_pic_eins)
title('debug usage')
%print('-f19','for_debugging_KLM_01','-dpng')

%use Gaussian filter
%________________________________
%adjust blur here
%remember to adjust the one on the lower part
actual_thres1 = uint8(after_thres1);
actual_thres2 = uint8(imgaussfilt(after_thres2, 0.5));
actual_thres3 = uint8(imgaussfilt(after_thres3, 1));
actual_thres4 = uint8(imgaussfilt(after_thres4, 1.5));
actual_thres5 = uint8(imgaussfilt(after_thres5, 2));
actual_thres6 = uint8(imgaussfilt(after_thres6, 2.5));
actual_thres7 = uint8(imgaussfilt(after_thres7, 3));
actual_thres8 = uint8(imgaussfilt(after_thres8, 3.5));
actual_thres9 = uint8(imgaussfilt(after_thres9, 4));
actual_thres10 = uint8(imgaussfilt(after_thres10, 4.5));
%________________________________

cd attempt_1_with_final_filter
%draw the picture
%attempt 1, plus them together 
res_pic_eins = zeros(size_pic_x, size_pic_y, 3); 
res_pic_eins = uint8(res_pic_eins); 
actual_thres = {actual_thres1, actual_thres2, actual_thres3, ... 
    actual_thres4, actual_thres5, actual_thres6, actual_thres7, ...
    actual_thres8, actual_thres9, actual_thres10};
for i = 1: 10
    res_pic_eins = res_pic_eins + actual_thres{i}; 
end 
res_pic_eins = uint8(res_pic_eins); 
new_res_pic_G = imgaussfilt(res_pic_eins, 2);
%attempt 2, replace the original picture with the number 
%if it's not black, replace it 
actual_thres = {actual_thres1, actual_thres2, actual_thres3, ... 
    actual_thres4, actual_thres5, actual_thres6, actual_thres7, ...
    actual_thres8, actual_thres9, actual_thres10};
actual_thres_gray = {rgb2gray(actual_thres1), rgb2gray(actual_thres2), ... 
    rgb2gray(actual_thres3), rgb2gray(actual_thres4), ...
    rgb2gray(actual_thres5), rgb2gray(actual_thres6), ...
    rgb2gray(actual_thres7), rgb2gray(actual_thres8), ...
    rgb2gray(actual_thres9), rgb2gray(actual_thres10)};
ad_pic = pic_case;
check_ad_pic = pic_case; 
for i = 1:10
    temp_pic = actual_thres_gray{i}; 
    draw = actual_thres{i}; 
    for x = 1: size_pic_x
        for y = 1: size_pic_y
            if temp_pic(x, y) > 20
                ad_pic(x, y, :) = draw(x, y, :); 
            end 
        end 
    end 
end
%As there's some werido, try to adjust it by Gaussian.
for i = 1:3
    new_res_pic_M(:, :, i) = medfilt2(res_pic_eins(:, :, i), [8,8]);
end 
new_res_pic_M_sharpened = imsharpen(new_res_pic_M,'Radius',2,'Amount',1); 
%figure(20)
subplot(2, 2, 1)
imshow(res_pic_eins)
title('final without adjustment')
subplot(2, 2, 2)
imshow(new_res_pic_G)
title('final with Gaussian blur adjustment')
subplot(2, 2, 3)
imshow(new_res_pic_M)
title('final with Medians blur adjustment')
subplot(2, 2, 4)
imshow(new_res_pic_M_sharpened)
title('final with Medians blur and sharpened adjustment')
%print('-f20','threshold_attempt_and_blur_comparision','-dpng')

%figure(21)
imshow(new_res_pic_G)
title('final with Gaussian blur adjustment')
%print('-f21','Gaussian_blur','-dpng')

%figure(22)
imshow(new_res_pic_M)
title('final with Medians blur adjustment')
%print('-f22','medians_blur','-dpng')

%figure(23)
imshow(new_res_pic_M_sharpened)
title('final with Medians blur and sharpened adjustment')
%print('-f23','medians_blur_and_sharpened','-dpng')

%figure(24)
imshow(res_pic_eins)
title('final without adjustment')
%print('-f24','no_adjust','-dpng')


%figure(25)
nbd = edge(rgb2gray(new_res_pic_M_sharpened), 'Roberts'); 
imshow(nbd)
%print('-f25','roberts_bd','-dpng')

%figure (26)
answ = rgb2gray(res_pic_eins); 
answ = double(answ); 
mesh(answ), 
ix = find(imregionalmax(answ));
%print('-f26','bd','-dpng')

%trying to fix the picture by readjusting #1 
%feels like putting correct dot back to correct place. 
cd ..
cd attempt_2_with_adjustion
%figure(27)
imshow(ad_pic)
title('second approach for picture')
%print('-f27','2nd_aph','-dpng')

%for debugging
pic_check = check_ad_pic - ad_pic; 
%figure(28)
imshow(pic_check)
title('checking')
%print('-f28','original_comp','-dpng')
pic_check_gray = rgb2gray(pic_check);
%figure(29)
mesh(pic_check_gray)
title('checking to gray')
%print('-f29','original_comp2gray_3D','-dpng')
%figure(30)
imshow(~pic_check_gray)
title('checking to gray')
%print('-f30','original_comp2gray','-dpng')

test_ad_pic = ad_pic; 
rgb_ad_pic = rgb2gray(ad_pic); 
temp_pic_case = rgb2gray(pic_case); 
count = 0; 

for x = size_pic_x
    for y = size_pic_y
        if rgb_ad_pic(x, y) > 30
            count = count + 1; 
            test_ad_pic (x, y, :) = pic_case(x, y, :); 
        end 
    end
end 
%figure(31)
subplot(1,3,1)
imshow(pic_case)
title('original')
subplot(1,3,2)
imshow(test_ad_pic)
title('trying to fix it')
subplot(1,3,3)
imshow(rgb2gray(pic_case - test_ad_pic))
title('difference')
%print('-f31','try_2','-dpng')

%figure(32)
mesh(rgb2gray(pic_case - test_ad_pic))
title('difference')
%print('-f32','comp_3D_diff','-dpng')
%trying to fix the picture by readjusting #2 
test_ad_pic = ad_pic; 
rgb_ad_pic = rgb2gray(ad_pic); 
temp_pic_case = rgb2gray(pic_case); 
redraw_ad = ad_pic; 
filtered_difference = zeros(size_pic_x, size_pic_y); 
counter_filtered = 0; 
for x = 1: size_pic_x
    for y = 1: size_pic_y
        if rgb2gray(pic_case(x, y, :) - test_ad_pic(x, y, :))  > 20
            counter_filtered  = counter_filtered  + 1; 
            filtered_difference(x, y) = rgb2gray(pic_case(x, y, :) - test_ad_pic(x, y, :)); 
            redraw_ad(x, y, :) = pic_case(x, y, :); 
        end 
    end 
end 
%figure(33)
mesh(filtered_difference)
title('filtered difference')
%print('-f33','filtered_difference','-dpng')
%figure(34)
imshow(redraw_ad)
title('redraw the picture')
nd = pic_case - redraw_ad; 
gray_nd = uint8(rgb2gray(nd));
%print('-f34','redraw_pic','-dpng')
%figure(35)
mesh(gray_nd)
title('redraw the picture and original difference')
%print('-f35','comp2_original_3D','-dpng')

%adjust it more precisely
nd_redraw_ad = redraw_ad; 
for x = 1: size_pic_x
    for y = 1: size_pic_y
        if gray_nd(x, y) > 5 
            nd_redraw_ad(x, y, :) = pic_case(x, y, :); 
        end 
    end 
end 
nd_1 = pic_case - nd_redraw_ad; 
gray_nd_1 = uint8(rgb2gray(nd_1));
%figure(36)
mesh(gray_nd_1)
title('redraw the picture and original difference')
%print('-f36','comp2_original_precise_3D','-dpng')

%-----------------------------------------------------------
redraw_1 = redraw_ad; 
%fill up the weird hole 
%use half and half as method
th_h1 = (th1+th2)/2;
th_h2 = (th2+th3)/2;
th_h3 = (th3+th4)/2;
th_h4 = (th4+th5)/2;
th_h5 = (th5+th6)/2;
th_h6 = (th6+th7)/2;
th_h7 = (th7+th8)/2;
th_h8 = (th8+th9)/2;
%use Gaussian filter to fill up holes 
actual_alln_thres1 = uint8(imgaussfilt(pic_case, 0.25));
actual_alln_thres2 = uint8(imgaussfilt(pic_case, 0.75));
actual_alln_thres3 = uint8(imgaussfilt(pic_case, 1.5-0.25));
actual_alln_thres4 = uint8(imgaussfilt(pic_case, 2-0.25));
actual_alln_thres5 = uint8(imgaussfilt(pic_case, 2.5-0.25));
actual_alln_thres6 = uint8(imgaussfilt(pic_case, 3-0.25));
actual_alln_thres7 = uint8(imgaussfilt(pic_case, 3.5-0.25));
actual_alln_thres8 = uint8(imgaussfilt(pic_case, 4-0.25));
actual_alln_thres9 = uint8(imgaussfilt(pic_case, 4.5-0.25));

distance_pos = {max_dis_sqrt, th1, th2, th3, th4, th5, th6, th7, th8, th9};
neue_thres = {actual_alln_thres1, actual_alln_thres2, actual_alln_thres3 ...
    actual_alln_thres4, actual_alln_thres5, actual_alln_thres6, ...
    actual_alln_thres7, actual_alln_thres8, actual_alln_thres9 }; 
for t = 1: 3
    for x = 1: size_pic_x
        for y = 1: size_pic_y
            if dis_pic_sqrt(x, y) < (max_dis_sqrt+th1)/2
                temp_thres = neue_thres{1}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) >= (max_dis_sqrt+th1)/2 && dis_pic_sqrt(x, y) < (th1+th2)/2
                temp_thres = neue_thres{2}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th1+th2)/2 && dis_pic_sqrt(x, y) < (th2+th3)/2
                temp_thres = neue_thres{3}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th2+th3)/2 && dis_pic_sqrt(x, y) < (th3+th4)/2
                temp_thres = neue_thres{4}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th3+th4)/2 && dis_pic_sqrt(x, y) < (th4+th5)/2
                temp_thres = neue_thres{5}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th4+th5)/2 && dis_pic_sqrt(x, y) < (th5+th6)/2
                temp_thres = neue_thres{6}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th5+th6)/2 && dis_pic_sqrt(x, y) < (th6+th7)/2
                temp_thres = neue_thres{7}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th6+th7)/2 && dis_pic_sqrt(x, y) < (th7+th8)/2
                temp_thres = neue_thres{8}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            elseif dis_pic_sqrt(x, y) > (th7+th8)/2 && dis_pic_sqrt(x, y) < (th8+th9)/2
                 temp_thres = neue_thres{9}; 
                if gray_nd_1(x, y) == 0
                    redraw_1(x, y, t) = temp_thres(x, y, t);
                end 
            end        
        end 
    end 
end 
figure(37)
imshow(redraw_1)
title('final result')
print('-f37','final','-dpng')
cd ..
cd ..
cd ..
cd ..
cd main
imwrite(redraw_1, 'final_pic.png');
imwrite(temp_pic_original, 'original_pic.png');
%----------------------------------------------------------------
%total written time: 20hrs 