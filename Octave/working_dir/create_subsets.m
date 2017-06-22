function [Xtrain Xval Xtest ytrain yval ytest] = create_subsets(X,y) 
  
 [m m_feat]=size(X);
 
 m_train=ceil(m*0.6);
 m_test=floor((m-m_train)/2);
 m_cv=m -m_train- m_test
 
 
 
 
 
 
 
 rand_rows=randperm(m);
 
 %Create Xtrain, Xval, Xtest of the set X 
 Xtrain=X(rand_rows(1:m_train),:);
 Xval=X(rand_rows(m_train+1:m_train+m_cv),:);
 Xtest=X(rand_rows(m-m_test+1:end),:);
 
 
 
 
 
 ytrain=y(rand_rows(1:m_train),:);
 yval=y(rand_rows(m_train+1:m_train+m_cv),:);
 ytest=y(rand_rows(m-m_test+1:end),:);

  
end