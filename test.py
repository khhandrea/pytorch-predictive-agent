from colorsys import hsv_to_rgb
import sys

import pygame

def draw_rainbow_rect(rect_x, rect_y, rect_width, rect_height):
    for x in range(rect_width):
        # x에 따라 hue 값 계산
        hue = x / rect_width
        rgb_color = [int(c * 255) for c in hsv_to_rgb(hue, 1, 1)]

        # 직사각형 그리기
        pygame.draw.rect(screen, rgb_color, (rect_x + x, 
                                            rect_y, 
                                            1, 
                                            rect_height))

if __name__ == '__main__':
    # Pygame 초기화
    pygame.init()

    # 화면 크기 설정
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pygame Example")

    # 직사각형 설정
    RECT_WIDTH = 360
    RECT_HEIGHT = 64
    rect_x = (screen_width - RECT_WIDTH) // 2
    rect_y = (screen_height - RECT_HEIGHT) // 2
    BORDER_COLOR = (0, 0, 0)
    BORDER_WIDTH = 2

    # 초기 색상 설정
    hue = 0
    color = pygame.Color(0, 0, 0)

    # 색상 변화 속도
    color_change_speed = 1

    # 캐릭터 설정
    character_size = 30
    character_x = (screen_width - character_size) // 2
    character_y = rect_y + RECT_HEIGHT + 48

    # 캐릭터 이동 속도
    character_speed = 5

    # 게임 루프
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    character_x -= character_speed
                elif event.key == pygame.K_RIGHT:
                    character_x += character_speed

        # 화면 그리기
        screen.fill((255, 255, 255))  # 화면을 흰색으로 지우기
        pygame.draw.rect(screen, BORDER_COLOR, (rect_x - BORDER_WIDTH,
                                                rect_y - BORDER_WIDTH,
                                                RECT_WIDTH + 2 * BORDER_WIDTH,
                                                RECT_HEIGHT + 2 * BORDER_WIDTH))  # 검은색 테두리 그리기
        draw_rainbow_rect(rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT) # 직사각형 그리기
        pygame.draw.polygon(screen, (0, 0, 255), [(character_x, character_y),
                                                (character_x + character_size, character_y),
                                                (character_x + character_size // 2, character_y - character_size)])  # 캐릭터 그리기

        # 화면 업데이트
        pygame.display.flip()

        # 초당 프레임 제한
        clock.tick(60)
