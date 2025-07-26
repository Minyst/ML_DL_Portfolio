import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'main_page.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  // Flutter 바인딩 초기화
  WidgetsFlutterBinding.ensureInitialized();

  // 세로 모드 고정
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // 상태바 스타일 설정
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Colors.black,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  try {
    // 사용 가능한 카메라 목록 가져오기
    cameras = await availableCameras();
    print('📷 사용 가능한 카메라: ${cameras.length}개');

    for (int i = 0; i < cameras.length; i++) {
      print('   카메라 $i: ${cameras[i].name} (${cameras[i].lensDirection})');
    }
  } catch (e) {
    print('❌ 카메라 초기화 실패: $e');
    cameras = [];
  }

  runApp(const SmartRecyclingApp());
}

class SmartRecyclingApp extends StatelessWidget {
  const SmartRecyclingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Recycling',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        // 다크 테마 기반
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: Colors.black,

        // AppBar 테마
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          iconTheme: IconThemeData(color: Colors.white),
          titleTextStyle: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.w600,
          ),
        ),

        // 버튼 테마
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF4CAF50),
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          ),
        ),

        // 텍스트 테마
        textTheme: const TextTheme(
          headlineLarge: TextStyle(
            color: Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.bold,
          ),
          headlineMedium: TextStyle(
            color: Colors.white,
            fontSize: 24,
            fontWeight: FontWeight.w600,
          ),
          bodyLarge: TextStyle(
            color: Colors.white,
            fontSize: 16,
          ),
          bodyMedium: TextStyle(
            color: Colors.white70,
            fontSize: 14,
          ),
        ),

        // 카드 테마
        cardTheme: const CardThemeData(
          color: Color(0xFF1E1E1E),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
          elevation: 8,
        ),

        // 아이콘 테마
        iconTheme: const IconThemeData(
          color: Colors.white,
          size: 24,
        ),
      ),

      home: cameras.isEmpty
          ? const CameraErrorScreen()
          : SmartRecyclingMainPage(cameras: cameras),
    );
  }
}

class CameraErrorScreen extends StatelessWidget {
  const CameraErrorScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.camera_alt_outlined,
                size: 80,
                color: Colors.grey,
              ),
              const SizedBox(height: 24),
              Text(
                '카메라 접근 오류',
                style: Theme.of(context).textTheme.headlineMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                '카메라에 접근할 수 없습니다.\n앱 권한을 확인해주세요.',
                style: Theme.of(context).textTheme.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: () async {
                  // 앱 재시작
                  try {
                    cameras = await availableCameras();
                    if (cameras.isNotEmpty) {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(
                          builder: (context) => SmartRecyclingMainPage(cameras: cameras),
                        ),
                      );
                    }
                  } catch (e) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text('카메라 초기화 실패: $e'),
                        backgroundColor: Colors.red,
                      ),
                    );
                  }
                },
                icon: const Icon(Icons.refresh),
                label: const Text('다시 시도'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// 글로벌 유틸리티 함수들
class AppConstants {
  // API 설정
  static const String apiBaseUrl = 'https://ml-dl-portfolio.onrender.com';
  static const String predictEndpoint = '/predict';

  // 앱 설정
  static const double maxImageSize = 1024; // 최대 이미지 크기
  static const int jpegQuality = 85; // JPEG 품질
  static const Duration requestTimeout = Duration(seconds: 30);

  // 전처리 설정
  static const double centerCropRatio = 0.85; // 중심 크롭 비율
  static const int edgeFadeWidth = 40; // 가장자리 페이드 너비
  static const double brightnessAdjustment = 0.1; // 밝기 조정값
  static const double contrastAdjustment = 1.2; // 대비 조정값

  // 색상
  static const Color primaryGreen = Color(0xFF4CAF50);
  static const Color secondaryGreen = Color(0xFF2E7D32);
  static const Color accentGreen = Color(0xFF81C784);
  static const Color errorRed = Color(0xFFE53E3E);
  static const Color warningOrange = Color(0xFFFF9800);
}

// 디버그 헬퍼
class DebugHelper {
  static bool get isDebugMode {
    bool debugMode = false;
    assert(debugMode = true);
    return debugMode;
  }

  static void log(String message) {
    if (isDebugMode) {
      print('🔍 [DEBUG] $message');
    }
  }

  static void logError(String message, [dynamic error]) {
    print('❌ [ERROR] $message');
    if (error != null) {
      print('   Details: $error');
    }
  }

  static void logSuccess(String message) {
    if (isDebugMode) {
      print('✅ [SUCCESS] $message');
    }
  }
}