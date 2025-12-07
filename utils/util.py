def pth_loader():
    global global_model_wrapper
    
    print("🚀 pths 로딩 중...")
    

    global_encoder.load_state_dict(torch.load("models/encoder_stage1.pth"))
    global_projector.load_state_dict(torch.load("models/projector_stage2.pth"))
    print("✅ pths 로드 완료.")


    print("🚀 pth opt 중...")
    full_model = SimCSEModelWrapper(global_encoder, global_projector)
    print("✅ pth 준비 완료.")
    
    full_model.to(DEVICE)
    full_model.eval() # 추론 모드

    global_model_wrapper = full_model
    print("✅ Full SimCSE Wrapper Loaded (Encoder + Projector)")
    
    return global_model_wrapper
