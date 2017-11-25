from kickstarter import *
df = process_train()
X,Y = pre_process_train(df)
train_x, train_y, test_x, test_y = split_in_test_and_train(X,Y)
accuracy = train_neural_network(train_x,train_y,test_x,test_y)
df_test = process_test()
X_test = pre_process_test(df_test)
prediction = test_neural_network(X_test)
rows_to_delete = ['goal','disable_communication',
					'country','currency','deadline','state_changed_at','created_at',
					'launched_at']
for i in rows_to_delete:
	del df_test[i]

df_test['final_status'] = prediction

df_test.to_csv('result.csv',header = 'final_status',index_label = "project_id",encoding ='utf-8')
