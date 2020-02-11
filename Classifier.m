clear
%load feature and label
load feature_500sample_64graylevel.mat
data=feature;
%%
fprintf('1.Relief-f \n2.PCA\n');
choice1 = input('');
fprintf('classifiers:\n1、LR\n2、LDA\n3、QDA\n4、NB\n5、svm-ecoc\n');
choice2 =input('');
if(choice2>6 || choice2<1)
    return;
end

class(1)=sum(label==1);
class(2)=sum(label==2);
class(3)=sum(label==3);

%% 
data = zscore(data);
k=10; %10-fold 

rp =randperm(size(label,1));
label=label(rp);
data=data(rp,:);
%%
p = cvpartition(label,'KFold',k);
mp= [];
if(choice1==1)
    [ranking, w] = relieff( data, label, 20);
elseif(choice1==2)
     %[coeff,~,latent] = pca(data,'Centered',false);
     coeff = pca(data,'Centered',false);
else
    fprintf('exit、\n');
    return ;
end
precise=[];
countf=0;
conMats=[];
trainError=[];
for select =20:4:90
    countf=countf+1;
    if(choice1==1)
        X=data(:,ranking(1:select));
    else
        X=data*coeff(:,1:select);
    end
    tmpError=[];
    for i = 1:k
        X_train = double(X(p.training(i),:)) ;
        Y_train = double(label(p.training(i))) ; % labels: neg_class -1, pos_class +1        
        X_test = double( X(p.test(i),:) );
        Y_test = double(label(p.test(i)));
        % model
        if(choice2==1)
            LR = mnrfit(X_train, Y_train);
            pihat = mnrval(LR,X_test );
            [M,I] = max(pihat');clear M
            I =I';
        elseif(choice2==2)
            lda = fitcdiscr(X_train,Y_train);
            [I,score,cost]=predict(lda,X_test);
            tmpError = [tmpError;resubLoss(lda)];
        elseif(choice2==3)
            qda = fitcdiscr(X_train,Y_train, 'DiscrimType','quadratic');
            [I,score,cost]=predict(qda,X_test);
            tmpError = [tmpError;resubLoss(qda)];
        elseif(choice2==4)
            nbGau = fitcnb(X_train, Y_train);
            I = predict(nbGau,X_test);
        elseif(choice2==5)
            svmecoe = fitcecoc(X_train,Y_train);
            I = predict(svmecoe,X_test);
        else
            exit();
        end
        count = sum(I==Y_test);
      
        precise(countf,i) = count/size(Y_test,1);
    end
    trainError = [trainError;sum(tmpError)];
end
mp=sum(precise,2)/10;
trainError=trainError/10;
trainPreicse=1-trainError;
mp=mp';
