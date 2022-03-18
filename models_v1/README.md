MODEL FILE จะถูกบันทึกในรูปแบบสกุล hdf5 (.h5)
โดยจะมีการบันทึกทุกๆครั้งที่ Epochs นั้นๆมีค่า Accuracy เพิ่มขึ้นจากครั้งก่อนหน้า
ชื่อไฟล์มีความหมายดังนี้ model.{epoch}-{validation_accuracy}-{validation_loss}.h5
แนะนำให้ลองใช้งานตัวไฟล์ model.h5 หรือ model-best.h5 ซึ่งถูกบันทึกโดย WandbCallback

models_v1
ใช้ Network Architecture ตาม Speech Recognition Challenge
มีการเดิม Kernel_regularization ใน Conv2d และประกาศเลเยอร์รวม Activation Function
