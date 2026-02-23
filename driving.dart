import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'dart:async';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart'
    hide NotificationVisibility;

import 'drowsinessChecker.dart';
import 'math_quiz.dart';
import 'speech_service.dart';
import 'number_parser.dart';

class DrivingScreen extends StatefulWidget {
  const DrivingScreen({super.key});

  @override
  State<DrivingScreen> createState() => _DrivingScreenState();
}

class _DrivingScreenState extends State<DrivingScreen> {
  CameraController? _controller;
  bool _isInitialized = false;
  bool _isProcessing = false;
  int drowsyStatus = 0;
  int serverStatus = 0;
  bool _isOverlayMode = false;

  Timer? _timer;
  int _elapsedSeconds = 0;
  DateTime _lastServerSend = DateTime.now();

  static const int SEND_INTERVAL_MS = 300;

  final MathQuizEngine _quizEngine = MathQuizEngine();
  final SpeechService _speechService = SpeechService();
  bool _isQuizActive = false;
  Quiz? _currentQuiz;
  bool _isAlarmPlaying = false;
  final FlutterTts _tts = FlutterTts();
  late AudioPlayer _audioPlayer;
  late Source _alarmSource;

  @override
  void initState() {
    super.initState();
    _audioPlayer = AudioPlayer();
    _alarmSource = AssetSource('warning.mp3');
    _initForegroundTask();
    _initializeCamera();
    _initTts();

    // ==========================================
    // 오버레이 리스너
    // ==========================================
    FlutterOverlayWindow.overlayListener.listen((data) async {
      if (mounted) {
        setState(() => _isOverlayMode = false);
        
        // 알림 문구 변경
        await _updateForegroundTaskToMain();
      }
    });
  }
  // ==============================
  // Foreground Task 초기화 
  // ==============================
    void _initForegroundTask() {
      FlutterForegroundTask.init(
        androidNotificationOptions: AndroidNotificationOptions(
          channelId: 'xident_drowsy',
          channelName: 'X-IDENT 졸음 감지',
          channelDescription: '백그라운드에서 졸음 감지 실행 중',
          channelImportance: NotificationChannelImportance.LOW,
          priority: NotificationPriority.LOW,
        ),
        iosNotificationOptions: const IOSNotificationOptions(),
        foregroundTaskOptions: ForegroundTaskOptions(
          eventAction: ForegroundTaskEventAction.nothing(),
          autoRunOnBoot: false,
        ),
      );
    }
  // ==========================================
  // 앱 복귀 시 포그라운드 알림 문구 업데이트 함수
  // ==========================================
  Future<void> _updateForegroundTaskToMain() async {
    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.updateService(
        notificationTitle: 'X-IDENT 졸음 감지 실행 중',
        notificationText: '안전 운전 하세요! 앱 화면에서 감지 중입니다.',
      );
    } else {
      // 다시 시작
      await FlutterForegroundTask.startService(
        serviceId: 256,
        notificationTitle: 'X-IDENT 졸음 감지 실행 중',
        notificationText: '안전 운전 하세요!',
      );
    }
  }

  void _initTts() async {
    await _tts.setLanguage("ko-KR");
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
  }

  // ==============================
  // 오버레이 모드 관련
  // ==============================

  Future<void> _startOverlayMode() async {
    bool hasPermission = await FlutterOverlayWindow.isPermissionGranted();

    if (!hasPermission) {
      bool? granted = await FlutterOverlayWindow.requestPermission();
      if (granted != true) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('플로팅 모드를 위해 다른 앱 위에 표시 권한이 필요합니다.'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }
    }

    // Foreground Service 시작 → Flutter 엔진 백그라운드 유지
    await FlutterForegroundTask.startService(
      serviceId: 256,
      notificationTitle: 'X-IDENT 졸음 감지 중',
      notificationText: '플로팅 모드 실행 중 - 탭하면 앱으로 돌아갑니다',
    );

    await FlutterOverlayWindow.showOverlay(
      enableDrag: true,
      overlayTitle: "X-IDENT 졸음 감지 중",
      overlayContent: "탭하면 앱으로 돌아갑니다",
      flag: OverlayFlag.defaultFlag,
      visibility: NotificationVisibility.visibilityPublic,
      positionGravity: PositionGravity.auto,
      width: 200,
      height: 200,
    );

    if (mounted) setState(() => _isOverlayMode = true);
  }

  void _notifyOverlay(int status) {
    if (_isOverlayMode) {
      FlutterOverlayWindow.shareData({'drowsyStatus': status});
    }
  }

  // ==============================
  // 퀴즈 관련
  // ==============================

  void _startQuizSequence() async {
    if (_isQuizActive) return;
    _quizEngine.reset();
    _isQuizActive = true;
    await _playWarningSound();
    await Future.delayed(const Duration(seconds: 2));
    _nextQuizStep();
  }

  void _nextQuizStep() async {
    if (!mounted) return;
    if (_quizEngine.isFinished()) {
      await _tts.speak("정답을 모두 맞히셨습니다. 안전 운전 하세요.");
      setState(() {
        _isQuizActive = false;
        drowsyStatus = 0;
      });
      _notifyOverlay(0);
      return;
    }
    _currentQuiz = _quizEngine.generateQuiz();
    await _tts.speak(_currentQuiz!.questionText);
    Future.delayed(const Duration(milliseconds: 4000), () {
      _listenForAnswer();
    });
  }

  void _listenForAnswer() async {
    final result = await _speechService.listenAndGetResult();
    if (result == null) {
      await _tts.speak("잘 듣지 못했습니다. 다시 말씀해 주세요.${_currentQuiz!.questionText}");
      Future.delayed(const Duration(milliseconds: 2000), () => _nextQuizStep());
    } else {
      _handleVoiceAnswer(result.toString());
    }
  }

  void _handleVoiceAnswer(String spokenText) async {
    bool? isCorrect = _quizEngine.submitAnswer(spokenText, _currentQuiz!.answer);
    if (isCorrect == true) {
      await _tts.speak("정답입니다!");
      Future.delayed(const Duration(milliseconds: 2000), () => _nextQuizStep());
    } else {
      await _tts.speak("오답입니다. 다시 계산해 보세요. ${_currentQuiz!.questionText}");
      Future.delayed(const Duration(milliseconds: 2000), () => _listenForAnswer());
    }
    if (mounted) setState(() {});
  }

  // ==============================
  // 카메라 및 서버 통신
  // ==============================

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    CameraDescription selectedCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: kIsWeb ? ImageFormatGroup.jpeg : ImageFormatGroup.nv21,
    );

    try {
      await _controller!.initialize();
      _startTimer();

      if (kIsWeb) {
        _startWebCaptureTimer();
      } else {
        _controller!.startImageStream((CameraImage image) {
          if (!_isProcessing) _processCameraImage(image);
        });
      }

      if (mounted) setState(() => _isInitialized = true);
    } catch (e) {
      debugPrint("카메라 초기화 실패: $e");
    }
  }

  void _processCameraImage(CameraImage image) async {
    if (_isProcessing || !mounted || _controller == null) return;

    final now = DateTime.now();
    if (now.difference(_lastServerSend).inMilliseconds < SEND_INTERVAL_MS) return;

    _isProcessing = true;
    _lastServerSend = now;

    try {
      await _sendRawImageToServer(image);
    } catch (e) {
      debugPrint('처리 에러: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Future<void> _sendRawImageToServer(CameraImage image) async {
    try {
      const String serverUrl = "http://172.30.1.81:8000/analyze_raw";
      var request = http.MultipartRequest('POST', Uri.parse(serverUrl));

      request.files.add(http.MultipartFile.fromBytes(
        'y_plane',
        image.planes[0].bytes,
        filename: 'y_plane.bin',
      ));

      if (image.planes.length > 1) {
        request.files.add(http.MultipartFile.fromBytes(
          'uv_plane',
          image.planes[1].bytes,
          filename: 'uv_plane.bin',
        ));
      } else {
        request.files.add(http.MultipartFile.fromBytes(
          'uv_plane',
          Uint8List(0),
          filename: 'empty_uv.bin',
        ));
      }

      request.fields['width'] = image.width.toString();
      request.fields['height'] = image.height.toString();
      request.fields['format'] = 'nv21';
      request.fields['row_stride'] = image.planes[0].bytesPerRow.toString();

      var response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final data = json.decode(respStr);
        final int receivedStatus = data['status'];

        if (mounted) {
          setState(() {
            serverStatus = receivedStatus;
            if (!_isQuizActive) {
              drowsyStatus = (receivedStatus == 2) ? 1 : 0;
            }
          });

          _notifyOverlay(drowsyStatus);

          if (receivedStatus == 2 && !_isQuizActive) {
            _playWarningSound();
            Future.delayed(const Duration(milliseconds: 1000), () {
              _startQuizSequence();
            });
          }
        }
      }
    } catch (e) {
      debugPrint("서버 통신 에러: $e");
    }
  }

  DateTime _lastCaptureTime = DateTime.now();

  void _startWebCaptureTimer() {
    _timer = Timer.periodic(const Duration(milliseconds: SEND_INTERVAL_MS), (timer) async {
      if (_isProcessing || !mounted || _controller == null || !_controller!.value.isInitialized) return;

      final now = DateTime.now();
      if (now.difference(_lastCaptureTime).inMilliseconds < SEND_INTERVAL_MS) return;

      _isProcessing = true;
      _lastCaptureTime = now;

      try {
        XFile file = await _controller!.takePicture();
        Uint8List jpegBytes = await file.readAsBytes();
        await _sendJpegToServer(jpegBytes);
      } catch (e) {
        debugPrint("웹 캡처 에러: $e");
      } finally {
        _isProcessing = false;
      }
    });
  }

  Future<void> _sendJpegToServer(Uint8List jpegBytes) async {
    try {
      const String serverUrl = "http://localhost:8000/analyze_jpeg";
      var request = http.MultipartRequest('POST', Uri.parse(serverUrl));
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        jpegBytes,
        filename: 'frame.jpg',
      ));

      var response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final data = json.decode(respStr);
        final int receivedStatus = data['status']; // 서버에서 받은 Level (0, 1, 2, 3)

        if (mounted) {
          setState(() {
            serverStatus = receivedStatus;
            // UI 표시용 상태 업데이트 (0이면 정상, 그 외엔 위험 상태로 간주 가능)
            drowsyStatus = (receivedStatus > 0) ? 1 : 0;
          });

          // 오버레이 창에도 상세 레벨 전송 (필요 시 오버레이 UI 색상 변경 가능)
          _notifyOverlay(receivedStatus);

          // ==========================================
          // [사양서 준수] 레벨별 차등 액션 수행
          // ==========================================
          if (receivedStatus == 1) {
            // Level 1: 가벼운 경고 알림음 송출 
            _playShortWarning(); 
          } 
          else if (receivedStatus == 2) {
            // Level 2: 날카로운 경고음 및 "잠시 쉬었다가 운전하세요" 음성 안내 
            _playSharpWarningWithVoice();
          } 
          else if (receivedStatus == 3) {
            // Level 3: 최상위 솔루션 (문제 풀이 시퀀스) 작동 
            if (!_isQuizActive) {
              _startQuizSequence();
            }
          }
        }
      }
    } catch (e) {
      debugPrint("서버 통신 에러: $e");
    }
  }

  // ==============================
  // 레벨별 알람 함수 정의
  // ==============================

  // Level 1용 짧은 경고
  void _playShortWarning() async {
    if (_isAlarmPlaying) return;
    _isAlarmPlaying = true;
    await _audioPlayer.play(AssetSource('warning.mp3'), volume: 0.5);
    await Future.delayed(const Duration(seconds: 1));
    _isAlarmPlaying = false;
  }

  // Level 2용 음성 포함 경고
  void _playSharpWarningWithVoice() async {
    if (_isAlarmPlaying) return;
    _isAlarmPlaying = true;
    await _audioPlayer.play(AssetSource('warning.mp3'), volume: 1.0);
    await Future.delayed(const Duration(milliseconds: 1500));
    await _tts.speak("잠시 쉬었다가 운전하세요."); // 사양서 지정 문구 
    _isAlarmPlaying = false;
  }

  // ==============================
  // 알람 및 기타 기능
  // ==============================

  Future<void> _playWarningSound() async {
    if (!mounted || _isAlarmPlaying) return;
    try {
      _isAlarmPlaying = true;
      if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
        await AudioPlayer.global.setAudioContext(AudioContext(
          android: const AudioContextAndroid(
            isSpeakerphoneOn: true,
            stayAwake: true,
            contentType: AndroidContentType.sonification,
            usageType: AndroidUsageType.notificationEvent,
            audioFocus: AndroidAudioFocus.gainTransientMayDuck,
          ),
        ));
      }
      await _audioPlayer.stop();
      await _audioPlayer.play(_alarmSource, volume: 1.0);
      await Future.delayed(const Duration(milliseconds: 2500));
      _isAlarmPlaying = false;
    } catch (e) {
      debugPrint("재생 에러: $e");
      _isAlarmPlaying = false;
    }
  }

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _elapsedSeconds++);
    });
  }

  String _formatTime(int seconds) {
    final h = seconds ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    final s = seconds % 60;
    return '${h.toString().padLeft(2, '0')}:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  @override
  void dispose() {
    FlutterForegroundTask.stopService();
    _timer?.cancel();
    _controller?.dispose();
    _audioPlayer.dispose();
    _tts.stop();
    super.dispose();
  }

  // ==============================
  // UI 빌드 부분
  // ==============================

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final isWide = size.width >= 600 || size.width > size.height;
    return Scaffold(
      backgroundColor: Colors.black,
      body: _isInitialized
          ? (isWide ? _buildWideLayout(context) : _buildMobileLayout(context))
          : const Center(child: CircularProgressIndicator(color: Color(0xFF13EC13))),
    );
  }

  Widget _buildMobileLayout(BuildContext context) {
    return Stack(
      children: [
        _cameraPreview(),
        SafeArea(
          child: Column(
            children: [
              _buildHeader(context),
              const Spacer(),
              _buildScannerFrame(),
              const Spacer(),
              _buildInfoPanel(),
              _buildFloatingModeButton(),
              const SizedBox(height: 12),
              _buildStopButton(),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildWideLayout(BuildContext context) {
    return Row(
      children: [
        Expanded(flex: 2, child: _cameraPreview()),
        Expanded(
          flex: 1,
          child: Container(
            color: Colors.black54,
            child: SafeArea(
              child: Column(
                children: [
                  _buildHeader(context),
                  const SizedBox(height: 20),
                  _buildScannerFrame(),
                  const Spacer(),
                  _buildInfoPanel(),
                  _buildFloatingModeButton(),
                  const SizedBox(height: 12),
                  _buildStopButton(),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _cameraPreview() {
    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _controller!.value.previewSize!.height,
          height: _controller!.value.previewSize!.width,
          child: CameraPreview(_controller!),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _glassIconButton(Icons.chevron_left, () => Navigator.pop(context)),
          _glassTag(drowsyStatus == 1 ? '위험: 졸음 감지' : '정상 주행 중'),
          _glassIconButton(Icons.settings, () {}),
        ],
      ),
    );
  }

  Widget _buildScannerFrame() {
    Color frameColor = drowsyStatus == 1 ? Colors.red : const Color(0xFF13EC13);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 40),
      child: Container(
        width: double.infinity,
        height: 260,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(40),
          border: Border.all(
              color: frameColor.withOpacity(0.9),
              width: drowsyStatus == 1 ? 4 : 2),
        ),
        child: drowsyStatus == 1
            ? const Center(
                child: Icon(Icons.warning_amber_rounded,
                    color: Colors.red, size: 80))
            : null,
      ),
    );
  }

  Widget _buildInfoPanel() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          color: const Color(0xFF141914).withOpacity(0.75),
          borderRadius: BorderRadius.circular(24)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _infoItem('운전 시간', _formatTime(_elapsedSeconds)),
          _divider(),
          _infoItem('알림 횟수', '0', valueColor: const Color(0xFF13EC13)),
          _divider(),
          _infoItem('안전 확률',
              drowsyStatus == 1 ? '40%' : '99%',
              valueColor: drowsyStatus == 1 ? Colors.red : Colors.white),
        ],
      ),
    );
  }

  Widget _buildFloatingModeButton() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: GestureDetector(
        onTap: _isOverlayMode ? null : _startOverlayMode,
        child: Container(
          width: double.infinity,
          height: 54,
          decoration: BoxDecoration(
            color: _isOverlayMode
                ? Colors.white12
                : Colors.white.withOpacity(0.08),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: _isOverlayMode
                  ? Colors.white24
                  : const Color(0xFF13EC13).withOpacity(0.6),
              width: 1.5,
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.picture_in_picture_alt,
                color: _isOverlayMode ? Colors.white38 : const Color(0xFF13EC13),
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                _isOverlayMode ? '플로팅 모드 실행 중 (탭하여 복귀)' : '플로팅 모드로 전환',
                style: GoogleFonts.notoSansKr(
                  color: _isOverlayMode ? Colors.white38 : const Color(0xFF13EC13),
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStopButton() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 20),
      child: ElevatedButton(
        onPressed: () => Navigator.pop(context),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF13EC13),
          minimumSize: const Size(double.infinity, 70),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.stop_circle, color: Colors.black),
            const SizedBox(width: 10),
            Text('운전 완료',
                style: GoogleFonts.notoSansKr(
                    color: Colors.black, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }

  Widget _glassIconButton(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: const BoxDecoration(color: Colors.white10, shape: BoxShape.circle),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _glassTag(String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
            color: drowsyStatus == 1 ? Colors.red : Colors.greenAccent),
      ),
      child: Text(label,
          style: TextStyle(
              color: drowsyStatus == 1 ? Colors.red : Colors.greenAccent,
              fontSize: 12,
              fontWeight: FontWeight.bold)),
    );
  }

  Widget _infoItem(String label, String value, {Color valueColor = Colors.white}) {
    return Column(children: [
      Text(label, style: const TextStyle(color: Colors.white38, fontSize: 9)),
      const SizedBox(height: 4),
      Text(value,
          style: GoogleFonts.notoSansKr(
              color: valueColor, fontSize: 16, fontWeight: FontWeight.bold)),
    ]);
  }

  Widget _divider() => Container(width: 1, height: 24, color: Colors.white10);
}