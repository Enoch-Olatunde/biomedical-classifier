clc; clear; close all;
pkg load statistics;
warning('off', 'all');

data = csvread('C:/Users/pc/PycharmProjects/PythonProject/breast_cancer.csv', 1, 0);
X = data(:, 1:end-1);
y = data(:, end);

rng(42);
n = rows(X);
idx = randperm(n);
n_train = floor(0.75*n);
X_train = X(idx(1:n_train), :); y_train = y(idx(1:n_train));
X_test = X(idx(n_train+1:end), :); y_test = y(idx(n_train+1:end));

b = glmfit(X_train, y_train, 'binomial', 'link', 'logit');
prob = glmval(b, X_test, 'logit');
y_pred = double(prob > 0.5);

accuracy = sum(y_pred == y_test) / length(y_test) * 100;
disp(['Accuracy: ', num2str(accuracy), '%'])

confusion = [sum((y_test==0) & (y_pred==0)), sum((y_test==0) & (y_pred==1));
             sum((y_test==1) & (y_pred==0)), sum((y_test==1) & (y_pred==1))];
disp('Confusion Matrix:')
disp(confusion)

figure;
imagesc(confusion);
colormap(flipud(gray));
colorbar;
title(['Confusion Matrix - Accuracy: ', num2str(accuracy), '%']);
xlabel('Predicted Label');
ylabel('True Label');
set(gca, 'xtick', [1 2], 'xticklabel', {'Malignant (0)', 'Benign (1)'});
set(gca, 'ytick', [1 2], 'yticklabel', {'Malignant (0)', 'Benign (1)'});

for i = 1:2
  for j = 1:2
    if confusion(i,j) > 40
      txtcolor = 'white';
    else
      txtcolor = 'black';
    end
    text(j, i, num2str(confusion(i,j)), 'HorizontalAlignment', 'center', 'FontSize', 22, 'FontWeight', 'bold', 'Color', txtcolor);
  end
end


