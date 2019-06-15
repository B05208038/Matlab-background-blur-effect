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

clear all, clc
pic_case = imread('image_0160.jpg'); 
gray_pic = rgb2gray(pic_case); 
[size_pic_x, size_pic_y] = size(gray_pic); 
R_shell_pic = uint8(pic_case(:, :, 1));
G_shell_pic = uint8(pic_case(:, :, 2));
B_shell_pic = uint8(pic_case(:, :, 3)); 
% compare colors
%{
figure(1)
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
%}

%{
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
%{
figure (2) 
imshow (output)
title ('gray with threshold')
%}
%}

%{
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

%{
no_bin = im2bw(output_RGB, 0.5);
acht_bin = im2bw(output_RGB, 0.8);
figure (3)
subplot(1, 3 ,1)
imshow(output_RGB)
title('threshold')
subplot(1, 3, 2)
imshow(no_bin)
title('bw, no cmap')
subplot(1, 3, 3)
imshow(acht_bin)
title('bw, cmap')
%}

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
figure(4)
subplot(1, 3, 1)
imshow(pic_case)
title('original picture'); 
subplot(1, 3, 2)
imshow(pixel_labels_original,[])
title('Image Labeled by Cluster Index--original');
subplot(1, 3, 3)
imshow(pixel_labels_output,[])
title('Image Labeled by Cluster Index--output');

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
figure (10)
imshow (~LAB_original_output, [])
%}
%write center into the for loop and pin it 
%take advantage of the rectangle to find the center of the face. 
body_detect = vision.CascadeObjectDetector('UpperBody'); 
BD_detect = step(body_detect, pic_case); 
center_face = zeros(size(BD_detect, 1), 2); 
for i = 1: size(BD_detect, 1) 
    center_face(i, 1) = BD_detect(i, 1)+BD_detect(i, 3)/2; 
    center_face(i, 2) = BD_detect(i, 2)+BD_detect(i, 4)/2;     
end
sobel_bw = edge(gray_pic, 'approxcanny');

figure(1)
imshow(sobel_bw); hold on 
for i = 1: size(BD_detect, 1)
    rectangle ('position', BD_detect(i, :), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
    plot(center_face(i, 1), center_face(i, 2),'.r','MarkerSize',30);
end 
title ('used trained cascade to detect face, example 1'); 

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

figure(2)
imshow(sobel_bw); hold on 
for i = 1: size(BD_NS, 1)
    rectangle ('position', actual_face(i, :), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'g'); 
    plot(center_nose(i, 1), center_nose(i, 2),'.b','MarkerSize',30);
    plot(center_face(i, 1), center_face(i, 2),'.r','MarkerSize',30);
end 
title ('confirm real face detecting, example 1'); 
hold off

%////////////////////
%with the method above, one can detect the real face and the center of the face. 
%The next step is to blur it and adjust it. 

%{
%Normally for a background substraction, one should have the reference
%background. But as we are doing this with a single picture, we will try to
%use color based segmentation to substract it. s
%{
figure(3)
imshow(pic_case)
%}
LAB_pic = rgb2lab(pic_case); 
ab = LAB_pic(:,:,2:3);
ab = im2single(ab);
nColors = 4;
% repeat the clustering 4 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
figure(4)
imshow(pixel_labels,[])
title('Image Labeled by Cluster Index');


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

%{
figure(5)
imshow(cluster1)
title('Objects in Cluster 1');
figure(6)
imshow(cluster2)
title('Objects in Cluster 2');
figure(7)
imshow(cluster3)
title('Objects in Cluster 3');
figure(8)
imshow(cluster4)
title('Objects in Cluster 4');
%}


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
figure(9)
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
%}

%reach 2, via KLT algorithm, one can find the eigenpoints of the face. 
%Use it as the edge, draw several circles to blurr it with different 
%magnitutde. 
%ref: https://www.mathworks.com/help/vision/examples/face-detection-and-tracking-using-the-klt-algorithm.html
points = detectMinEigenFeatures(rgb2gray(pic_case), 'ROI', BD_detect);
% Display the detected points.
figure(30), imshow(pic_case), hold on, title('Detected features');
plot(points);
position = points.Location; 
[pt_number, temp_num] = size(position); 
counter = zeros(pt_number, 1); 
for i = 1: pt_number
    counter(i) = abs(center_face(1)-position(i, 1)).^2+abs(center_face(2)-position(i, 2)).^2;
end
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
figure (4)
mesh(dis_pic_sqrt)
title('find the center of picture')
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
figure(5)
mesh(threshold_pic)
title('threshold_pic')

%for debugging issue 
temp_pic = zeros(size_pic_x, size_pic_y); 
temp_pic_eins = zeros(size_pic_x, size_pic_y, 3); 
for i = 1: 10
    temp_pic = temp_pic + processing_pic(:, :, i); 
end 
temp_pic_eins = temp_pic_eins+ after_thres1; 
%temp_pic_eins = temp_pic_eins + after_thres2+after_thres3+after_thres4;
%temp_pic_eins = temp_pic_eins+ after_thres5+after_thres6+after_thres7; 
%temp_pic_eins = temp_pic_eins+ after_thres8+after_thres9+after_thres10;
temp_pic_eins = temp_pic_eins ./ 256; 
figure(6)
mesh(temp_pic)
title('temp_pic')
figure(7)
imshow(temp_pic_eins)
title('debug usage')

%use Gaussian filter
actual_thres1 = after_thres1;
actual_thres2 = imgaussfilt(after_thres2, 0.5);
actual_thres3 = imgaussfilt(after_thres3, 1);
actual_thres4 = imgaussfilt(after_thres4, 1.5);
actual_thres5 = imgaussfilt(after_thres5, 2);
actual_thres6 = imgaussfilt(after_thres6, 2.5);
actual_thres7 = imgaussfilt(after_thres7, 3);
actual_thres8 = imgaussfilt(after_thres8, 3.5);
actual_thres9 = imgaussfilt(after_thres9, 4);
actual_thres10 = imgaussfilt(after_thres10, 4.5);

%draw the picture
res_pic_eins = zeros(size_pic_x, size_pic_y, 3); 
res_pic_eins = res_pic_eins+ actual_thres1; 
res_pic_eins = res_pic_eins + actual_thres2+actual_thres3+actual_thres4;
res_pic_eins = res_pic_eins+ actual_thres5+actual_thres6+actual_thres7; 
res_pic_eins = res_pic_eins+ actual_thres8+actual_thres9+actual_thres10;
res_pic_eins = res_pic_eins ./ 256; 
new_res_pic = imgaussfilt(res_pic_eins, 2);
figure(8)
subplot(2, 1, 1)
imshow(res_pic_eins)
title('final without adjustment')
subplot(2, 1, 2)
imshow(new_res_pic)
title('final with blur adjustment')