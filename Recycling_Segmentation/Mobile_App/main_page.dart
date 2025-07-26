import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'main.dart';

class SmartRecyclingMainPage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const SmartRecyclingMainPage({super.key, required this.cameras});

  @override
  State<SmartRecyclingMainPage> createState() => _SmartRecyclingMainPageState();
}

class _SmartRecyclingMainPageState extends State<SmartRecyclingMainPage>
    with WidgetsBindingObserver {

  // 카메라 관련
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isProcessing = false;

  // 결과 관련
  Uint8List? _resultOverlay;
  Uint8List? _resultPrediction;
  String? _processingStatus;

  // UI 상태
  bool _showGuide = true;
  double _processingProgress = 0.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  // ===== 카메라 초기화 =====

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      DebugHelper.logError('사용 가능한 카메라가 없습니다');
      return;
    }

    try {
      // 후면 카메라 우선 선택
      final camera = widget.cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => widget.cameras.first,
      );

      _cameraController = CameraController(
        camera,
        ResolutionPreset.high, // 고화질로 촬영
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _cameraController!.initialize();

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
        DebugHelper.logSuccess('카메라 초기화 완료');
      }
    } catch (e) {
      DebugHelper.logError('카메라 초기화 실패', e);
    }
  }

  // ===== 스마트 이미지 전처리 =====

  Future<Uint8List> _preprocessImage(Uint8List imageBytes) async {
    DebugHelper.log('이미지 전처리 시작...');

    try {
      // 1. 이미지 디코딩
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) throw Exception('이미지 디코딩 실패');

      DebugHelper.log('원본 크기: ${image.width}x${image.height}');

      // 2. 크기 조정 (처리 속도 향상)
      if (image.width > AppConstants.maxImageSize || image.height > AppConstants.maxImageSize) {
        double scale = AppConstants.maxImageSize / math.max(image.width, image.height);
        int newWidth = (image.width * scale).round();
        int newHeight = (image.height * scale).round();
        image = img.copyResize(image, width: newWidth, height: newHeight);
        DebugHelper.log('리사이즈 완료: ${image.width}x${image.height}');
      }

      // 3. 중심 크롭 (관심 영역에 집중)
      image = _centerCrop(image, AppConstants.centerCropRatio);
      DebugHelper.log('중심 크롭 완료: ${image.width}x${image.height}');

      // 4. 조명 정규화
      image = _normalizeImage(image);
      DebugHelper.log('조명 정규화 완료');

      // 5. 가장자리 페이드 (배경 최소화)
      image = _applyEdgeFade(image, AppConstants.edgeFadeWidth);
      DebugHelper.log('가장자리 페이드 완료');

      // 6. 배경 간소화
      image = _simplifyBackground(image);
      DebugHelper.log('배경 간소화 완료');

      // 7. 최종 인코딩
      final processedBytes = Uint8List.fromList(
          img.encodeJpg(image, quality: AppConstants.jpegQuality)
      );

      DebugHelper.logSuccess('전처리 완료: ${processedBytes.length} bytes');
      return processedBytes;

    } catch (e) {
      DebugHelper.logError('이미지 전처리 실패', e);
      rethrow;
    }
  }

  img.Image _centerCrop(img.Image image, double ratio) {
    int newWidth = (image.width * ratio).round();
    int newHeight = (image.height * ratio).round();
    int startX = (image.width - newWidth) ~/ 2;
    int startY = (image.height - newHeight) ~/ 2;

    return img.copyCrop(image,
        x: startX,
        y: startY,
        width: newWidth,
        height: newHeight
    );
  }

  img.Image _normalizeImage(img.Image image) {
    // 밝기와 대비 조정
    image = img.adjustColor(image,
      brightness: AppConstants.brightnessAdjustment,
      contrast: AppConstants.contrastAdjustment,
    );

    // 선명도 향상 (convolution 대신 다른 방법 사용)
    return img.gaussianBlur(image, radius: 1);
  }

  img.Image _applyEdgeFade(img.Image image, int fadeWidth) {
    // 가장자리를 서서히 흰색으로 페이드
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        int distanceFromEdge = math.min(
            math.min(x, image.width - x),
            math.min(y, image.height - y)
        );

        if (distanceFromEdge < fadeWidth) {
          double fadeRatio = distanceFromEdge / fadeWidth;
          img.Pixel pixel = image.getPixel(x, y);

          int r = (pixel.r * fadeRatio + 255 * (1 - fadeRatio)).round();
          int g = (pixel.g * fadeRatio + 255 * (1 - fadeRatio)).round();
          int b = (pixel.b * fadeRatio + 255 * (1 - fadeRatio)).round();

          image.setPixel(x, y, img.ColorRgb8(r, g, b));
        }
      }
    }
    return image;
  }

  img.Image _simplifyBackground(img.Image image) {
    // 중심 영역 외부의 복잡한 배경을 단순화
    int centerX = image.width ~/ 2;
    int centerY = image.height ~/ 2;
    int radius = math.min(image.width, image.height) ~/ 3;

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        double distance = math.sqrt(
            math.pow(x - centerX, 2) + math.pow(y - centerY, 2)
        );

        if (distance > radius) {
          // 중심에서 멀리 떨어진 픽셀은 블러 처리
          img.Pixel pixel = image.getPixel(x, y);
          int gray = ((pixel.r + pixel.g + pixel.b) / 3).round();

          // 약간의 색상을 유지하면서 단순화
          int r = (pixel.r * 0.3 + gray * 0.7).round();
          int g = (pixel.g * 0.3 + gray * 0.7).round();
          int b = (pixel.b * 0.3 + gray * 0.7).round();

          image.setPixel(x, y, img.ColorRgb8(r, g, b));
        }
      }
    }
    return image;
  }

  // ===== API 통신 =====

  Future<void> _sendToServer(Uint8List imageBytes) async {
    setState(() {
      _processingStatus = '서버로 전송 중...';
      _processingProgress = 0.3;
    });

    try {
      final uri = Uri.parse('${AppConstants.apiBaseUrl}${AppConstants.predictEndpoint}');
      final request = http.MultipartRequest('POST', uri);

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          imageBytes,
          filename: 'preprocessed_image.jpg',
        ),
      );

      setState(() {
        _processingStatus = 'AI 분석 중...';
        _processingProgress = 0.6;
      });

      final streamedResponse = await request.send().timeout(AppConstants.requestTimeout);
      final response = await http.Response.fromStream(streamedResponse);

      setState(() {
        _processingStatus = '결과 처리 중...';
        _processingProgress = 0.9;
      });

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);

        if (jsonData['status'] == 'success') {
          final overlayBase64 = jsonData['overlay'] as String;
          final predictionBase64 = jsonData['prediction'] as String;

          setState(() {
            _resultOverlay = base64Decode(overlayBase64);
            _resultPrediction = base64Decode(predictionBase64);
            _processingStatus = '완료!';
            _processingProgress = 1.0;
          });

          DebugHelper.logSuccess('AI 분석 완료');

          // 결과 화면으로 이동
          _showResultDialog();

        } else {
          throw Exception('서버 응답 오류: ${jsonData['message'] ?? 'Unknown error'}');
        }
      } else {
        throw Exception('HTTP ${response.statusCode}: ${response.body}');
      }

    } catch (e) {
      DebugHelper.logError('서버 통신 실패', e);
      _showErrorDialog('분석 실패: ${e.toString()}');
    } finally {
      setState(() {
        _isProcessing = false;
        _processingStatus = null;
        _processingProgress = 0.0;
      });
    }
  }

  // ===== 카메라 촬영 및 처리 =====

  Future<void> _captureAndProcess() async {
    if (!_isCameraInitialized || _isProcessing) return;

    setState(() {
      _isProcessing = true;
      _processingStatus = '사진 촬영 중...';
      _processingProgress = 0.1;
    });

    try {
      // 1. 사진 촬영
      final XFile image = await _cameraController!.takePicture();
      final imageBytes = await image.readAsBytes();

      DebugHelper.log('사진 촬영 완료: ${imageBytes.length} bytes');

      setState(() {
        _processingStatus = '이미지 전처리 중...';
        _processingProgress = 0.2;
      });

      // 2. 스마트 전처리
      final processedBytes = await _preprocessImage(imageBytes);

      // 3. 서버로 전송
      await _sendToServer(processedBytes);

    } catch (e) {
      DebugHelper.logError('촬영 및 처리 실패', e);
      _showErrorDialog('처리 실패: ${e.toString()}');
      setState(() {
        _isProcessing = false;
        _processingStatus = null;
        _processingProgress = 0.0;
      });
    }
  }

  // ===== UI 다이얼로그 =====

  void _showResultDialog() {
    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        child: Container(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.8,
          ),
          decoration: BoxDecoration(
            color: const Color(0xFF1E1E1E),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // 헤더
              Container(
                padding: const EdgeInsets.all(20),
                decoration: const BoxDecoration(
                  color: AppConstants.primaryGreen,
                  borderRadius: BorderRadius.only(
                    topLeft: Radius.circular(20),
                    topRight: Radius.circular(20),
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.recycling, color: Colors.white),
                    const SizedBox(width: 12),
                    const Text(
                      'AI 분석 결과',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Spacer(),
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.close, color: Colors.white),
                    ),
                  ],
                ),
              ),

              // 결과 이미지
              Flexible(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      if (_resultOverlay != null) ...[
                        Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.3),
                                blurRadius: 10,
                                offset: const Offset(0, 5),
                              ),
                            ],
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.memory(
                              _resultOverlay!,
                              fit: BoxFit.contain,
                            ),
                          ),
                        ),
                        const SizedBox(height: 20),
                      ],

                      // 액션 버튼들
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: () {
                                Navigator.pop(context);
                                _captureAndProcess();
                              },
                              icon: const Icon(Icons.camera_alt),
                              label: const Text('다시 촬영'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF424242),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: () => Navigator.pop(context),
                              icon: const Icon(Icons.check),
                              label: const Text('확인'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Row(
          children: [
            Icon(Icons.error_outline, color: AppConstants.errorRed),
            SizedBox(width: 8),
            Text('오류', style: TextStyle(color: Colors.white)),
          ],
        ),
        content: Text(
          message,
          style: const TextStyle(color: Colors.white70),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인', style: TextStyle(color: AppConstants.primaryGreen)),
          ),
        ],
      ),
    );
  }

  void _showGuideDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Row(
          children: [
            Icon(Icons.help_outline, color: AppConstants.primaryGreen),
            SizedBox(width: 8),
            Text('촬영 가이드', style: TextStyle(color: Colors.white)),
          ],
        ),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('📸 더 정확한 인식을 위해:', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            SizedBox(height: 12),
            Text('• 물체를 화면 중앙에 배치하세요', style: TextStyle(color: Colors.white70)),
            Text('• 깔끔한 배경에서 촬영하세요', style: TextStyle(color: Colors.white70)),
            Text('• 충분한 조명을 확보하세요', style: TextStyle(color: Colors.white70)),
            Text('• 물체가 선명하게 보이도록 하세요', style: TextStyle(color: Colors.white70)),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              setState(() => _showGuide = false);
            },
            child: const Text('다시 보지 않기', style: TextStyle(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인', style: TextStyle(color: AppConstants.primaryGreen)),
          ),
        ],
      ),
    );
  }

  // ===== UI 빌드 =====

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: const Text('스마트 재활용 인식'),
        actions: [
          IconButton(
            onPressed: _showGuideDialog,
            icon: const Icon(Icons.help_outline),
          ),
        ],
      ),
      body: Stack(
        children: [
          // 카메라 프리뷰
          if (_isCameraInitialized)
            Positioned.fill(
              child: CameraPreview(_cameraController!),
            )
          else
            const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: AppConstants.primaryGreen),
                  SizedBox(height: 16),
                  Text('카메라 초기화 중...', style: TextStyle(color: Colors.white)),
                ],
              ),
            ),

          // 가이드 오버레이
          if (_isCameraInitialized && _showGuide)
            Positioned.fill(
              child: Container(
                color: Colors.black.withOpacity(0.3),
                child: Center(
                  child: Container(
                    width: 280,
                    height: 280,
                    decoration: BoxDecoration(
                      border: Border.all(color: AppConstants.primaryGreen, width: 3),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Center(
                      child: Text(
                        '재활용품을\n이 영역에 배치하세요',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          shadows: [
                            Shadow(
                              offset: Offset(1, 1),
                              blurRadius: 3,
                              color: Colors.black,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),

          // 처리 중 오버레이
          if (_isProcessing)
            Positioned.fill(
              child: Container(
                color: Colors.black.withOpacity(0.8),
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E1E1E),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.psychology,
                          size: 64,
                          color: AppConstants.primaryGreen,
                        ),
                        const SizedBox(height: 20),
                        Text(
                          _processingStatus ?? 'AI 분석 중...',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 20),
                        SizedBox(
                          width: 200,
                          child: LinearProgressIndicator(
                            value: _processingProgress,
                            backgroundColor: Colors.grey[700],
                            valueColor: const AlwaysStoppedAnimation<Color>(
                              AppConstants.primaryGreen,
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${(_processingProgress * 100).round()}%',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 14,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

          // 하단 촬영 버튼
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _isProcessing ? null : _captureAndProcess,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: _isProcessing
                        ? Colors.grey
                        : AppConstants.primaryGreen,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 10,
                        offset: const Offset(0, 5),
                      ),
                    ],
                  ),
                  child: Icon(
                    _isProcessing ? Icons.hourglass_empty : Icons.camera_alt,
                    color: Colors.white,
                    size: 36,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}