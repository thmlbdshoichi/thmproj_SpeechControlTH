MODEL FILE จะถูกบันทึกในรูปแบบสกุล hdf5 (.h5)
โดยจะมีการบันทึกทุกๆครั้งที่ Epochs นั้นๆมีค่า Accuracy เพิ่มขึ้นจากครั้งก่อนหน้า
ชื่อไฟล์มีความหมายดังนี้ model.{epoch}-{validation_accuracy}-{validation_loss}.h5
แนะนำให้ลองใช้งานตัวไฟล์ model.h5 หรือ model-best.h5 ซึ่งถูกบันทึกโดย WandbCallback

# MODEL ตัวนี้เป็นเวอร์ชั่นปรับปรุง
มีการตัด Dropout ออกไป 2 Layer เพื่อให้เข้ากับการแปลงโมเดลของ Vitis-AI มากที่สุด
