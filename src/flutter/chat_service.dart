import 'dart:convert';
import 'package:http/http.dart' as http;

class ChatService {
  // Replace with your Lightning.ai Public URL
  // Example: "https://<your-app-name>.lightning.ai/chat"
  // Note: Ensure you append "/chat" to the base URL provided by Lightning.ai
  String baseUrl = "YOUR_LIGHTNING_PUBLIC_URL/chat";

  ChatService({String? customUrl}) {
    if (customUrl != null) {
      baseUrl = customUrl;
    }
  }

  /// Updates the base URL dynamically (e.g., after restarting Lightning Studio)
  void updateBaseUrl(String newUrl) {
    // Ensure the URL ends with /chat endpoint
    if (!newUrl.endsWith("/chat")) {
      baseUrl = "$newUrl/chat";
    } else {
      baseUrl = newUrl;
    }
  }

  /// Sends a question to the chatbot and returns the answer.
  /// Throws an exception if the request fails or times out.
  Future<String> askChatbot(String question) async {
    try {
      final uri = Uri.parse(baseUrl);
      
      final response = await http.post(
        uri,
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: jsonEncode({"question": question}),
      ).timeout(
        const Duration(seconds: 60), // LLMs can be slow, set generous timeout
        onTimeout: () {
          throw Exception("Request timed out. The model is taking too long to respond.");
        },
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data["answer"] ?? "No answer received.";
      } else {
        throw Exception("Failed to get answer. Status Code: ${response.statusCode}. Body: ${response.body}");
      }
    } catch (e) {
      throw Exception("Error communicating with chatbot: $e");
    }
  }
}
